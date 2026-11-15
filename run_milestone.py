"""Check the mechanism and reproduce the development-only milestone."""
import argparse
import json
import math
from pathlib import Path
import platform
import time

from check_starter import check_starter
from evaluate import evaluate
from nsc import capacity_components, layer_capacities, original_nsc, psi_mp, resource_estimate
from student_score import ALPHA, ALPHA_GRID, ablation, position_control, preceding_fractions, score


ROOT = Path(__file__).resolve().parent
STUDENT = ROOT / 'student_score.py'
DIAGNOSTIC_ALPHA = 1.0


def check_mechanism(configs):
    """Check limiting cases and two synthetic interventions before evaluating PPL."""
    first = {'d_model': 128, 'n_layer': 2, 'n_head': 4, 'd_inner': [256, 512]}
    reversed_ffn = dict(first, d_inner=[512, 256])
    single = dict(first, n_layer=1, d_inner=256)
    assert preceding_fractions([0.0, 5.0]) == [0.0, 0.0]
    assert preceding_fractions([3.0, 3.0]) == [0.0, 0.5]
    for alpha in ALPHA_GRID:
        assert score(single, alpha) == position_control(single, alpha) == original_nsc(single)
    assert original_nsc(first) == original_nsc(reversed_ffn)
    assert ablation(first) == ablation(reversed_ffn) == original_nsc(first)
    assert resource_estimate(first) == resource_estimate(reversed_ffn)
    assert score(first, DIAGNOSTIC_ALPHA) != score(reversed_ffn, DIAGNOSTIC_ALPHA)

    reorder = []
    for config in (first, reversed_ffn):
        reorder.append({
            'ffn_widths': config['d_inner'],
            'original_nsc': original_nsc(config),
            'score': score(config, DIAGNOSTIC_ALPHA),
            'position_control': position_control(config, DIAGNOSTIC_ALPHA),
            'ablation': ablation(config),
            **resource_estimate(config),
        })

    fixed_prefix = dict(first, n_layer=3, d_inner=[256, 512, 256])
    larger_suffix = dict(fixed_prefix, d_inner=[256, 512, 1024])
    suffix = []
    for config in (fixed_prefix, larger_suffix):
        capacities = layer_capacities(config)
        fraction = preceding_fractions(capacities)[1]
        attention, ffn = capacity_components(config)[1]
        suffix.append({
            'ffn_widths': config['d_inner'],
            'total_nsc': original_nsc(config),
            'layer2_prefix_nsc': capacities[0],
            'layer2_fraction': fraction,
            'layer2_ffn_multiplier': 1 / (1 + DIAGNOSTIC_ALPHA * fraction),
            'position_layer2_ffn_multiplier': 1 / (1 + DIAGNOSTIC_ALPHA / len(capacities)),
            'layer2_ffn_discounted': ffn / (1 + DIAGNOSTIC_ALPHA * fraction),
            'layer2_attention': attention,
        })
    assert suffix[0]['layer2_prefix_nsc'] == suffix[1]['layer2_prefix_nsc']
    assert suffix[0]['layer2_fraction'] > suffix[1]['layer2_fraction']
    assert suffix[0]['layer2_ffn_multiplier'] < suffix[1]['layer2_ffn_multiplier']
    assert suffix[0]['position_layer2_ffn_multiplier'] == suffix[1]['position_layer2_ffn_multiplier']

    for config in configs:
        reference = original_nsc(config)
        assert ablation(config) == position_control(config, 0.0) == reference
        for alpha in ALPHA_GRID:
            for method in (score, position_control):
                assert 0 < method(config, alpha) <= reference
    return {'alpha': DIAGNOSTIC_ALPHA, 'panel_checked': len(configs),
            'reorder': reorder, 'suffix_intervention': suffix}


def compare_starter(result):
    """Verify the alpha-zero evaluation against the supplied reference output."""
    expected = json.loads((ROOT / 'expected_development.json').read_text())
    assert result['ranking']['position_control'] == expected['ranking']['original_nsc']
    assert {name: result['ranking'][name] for name in expected['ranking']} == expected['ranking']
    assert len(result['selection']) == len(expected['selection'])
    for actual, reference in zip(result['selection'], expected['selection']):
        assert {name: actual[name] for name in reference if name != 'methods'} == {
            name: value for name, value in reference.items() if name != 'methods'
        }
        for name, metrics in reference['methods'].items():
            for metric, value in metrics.items():
                observed = actual['methods'][name][metric]
                # Across Python builds, a floating-point SD can differ by one ULP.
                assert observed == value or math.isclose(observed, value, abs_tol=1e-12, rel_tol=0)
        assert actual['methods']['position_control'] == actual['methods']['original_nsc']


def scoring_times(configs, alpha):
    """Time one cold-cache pass and its subsequent warm-cache pass per method."""
    methods = {
        'original_nsc': original_nsc,
        'student': lambda config: score(config, alpha),
        'position_control': lambda config: position_control(config, alpha),
    }
    timings = {}
    for name, method in methods.items():
        psi_mp.cache_clear()
        timings[name] = {}
        for cache in ('cold', 'warm'):
            start = time.perf_counter()
            for config in configs:
                method(config)
            timings[name][cache] = time.perf_counter() - start
    return timings


def run():
    """Tune with the fixed development grid and return results and diagnostics."""
    start = time.perf_counter()
    check_starter(development_only=True)
    data = ROOT / 'data'
    archive = json.loads((data / 'architectures.json').read_text())
    splits = json.loads((data / 'splits.json').read_text())
    labels = json.loads((data / 'development_labels.json').read_text())
    keys = splits['development']
    assert len(set(keys)) == len(keys) and set(labels) == set(keys)
    assert all(math.isfinite(labels[key]['valid_ppl']) and labels[key]['valid_ppl'] > 0 for key in keys)
    configs = [archive[key] for key in keys]
    mechanism = check_mechanism(configs)
    print(f'PASS: mechanism checks and exact baseline recovery on {len(configs)} development designs.')

    panels = {alpha: evaluate('development', STUDENT, alpha=alpha) for alpha in ALPHA_GRID}
    compare_starter(panels[0.0])
    print('PASS: alpha-zero evaluation matches the supplied development reference.')
    ranks = {
        alpha: {name: panel['ranking'][name] for name in ('student', 'position_control')}
        for alpha, panel in panels.items()
    }
    # Resolve equal Spearman values in favor of the smaller coefficient.
    best = {
        name: max(ALPHA_GRID, key=lambda alpha: (ranks[alpha][name]['spearman_rho'], -alpha))
        for name in ('student', 'position_control')
    }
    selected = best['student']
    assert ALPHA == selected, 'The default scorer must use the development-selected alpha.'
    default_panel = evaluate('development', STUDENT)
    for field in ('ranking', 'selection'):
        assert default_panel[field] == panels[selected][field]

    result = {
        'runtime': {
            'python': platform.python_version(),
            'platform': platform.platform(),
            'scoring_seconds': scoring_times(configs, selected),
            'wall_seconds': time.perf_counter() - start,
        },
        'checks': mechanism,
        'alpha_policy': 'maximize development Spearman rho; smallest alpha breaks exact ties',
        'same_alpha': selected,
        'best_development_alpha': best,
        'default_alpha': ALPHA,
        'development_grid': [{'alpha': alpha, 'ranking': ranks[alpha]} for alpha in ALPHA_GRID],
        'development': panels[selected],
    }
    return result


def main():
    """Write results to JSON and print the selected development coefficients."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'development_results.json')
    args = parser.parse_args()
    result = run()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f"Capacity alpha: {result['same_alpha']:g}; "
          f"independent position alpha: {result['best_development_alpha']['position_control']:g}.")
    print(f"Results: {args.output}")


if __name__ == '__main__':
    main()
