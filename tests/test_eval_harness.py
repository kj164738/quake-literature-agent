from quake_agent.eval_harness import load_cases, run_eval_cases


def test_eval_harness_passes_default_cases():
    cases = load_cases("eval_cases.json")

    results = run_eval_cases(cases)

    assert all(result.passed for result in results)
    assert {result.case_id for result in results} == {
        "local_early_warning",
        "force_arxiv_tool",
        "refuse_without_sources",
    }

