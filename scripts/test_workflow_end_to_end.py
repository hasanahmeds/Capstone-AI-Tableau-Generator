import pandas as pd
from workflow import DashboardGeneratorWorkflow
from schemas import WorkflowConfig


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def print_separator():
    print("=" * 80)


def print_test_header(test_name, purpose):
    print_separator()
    print(f"TEST: {test_name}")
    print("-" * 80)
    print(f"Purpose: {purpose}")
    print("-" * 80)


def print_result_summary(result):
    progress = result.get("progress", {})
    current_stage = progress.get("current_stage", "N/A")
    output_path = result.get("output_path", None)
    errors = result.get("errors", [])

    print(f"Final Stage      : {current_stage}")
    print(f"Output Path      : {output_path}")
    print(f"Error Count      : {len(errors)}")

    if result.get("dataset_schema") is not None:
        ds = result["dataset_schema"]
        row_count = ds.get("row_count", "N/A") if isinstance(ds, dict) else getattr(ds, "row_count", "N/A")
        col_count = ds.get("column_count", "N/A") if isinstance(ds, dict) else getattr(ds, "column_count", "N/A")
        print(f"Rows / Columns   : {row_count} / {col_count}")

    if result.get("quality_report") is not None:
        qr = result["quality_report"]
        quality_score = qr.get("quality_score", "N/A") if isinstance(qr, dict) else getattr(qr, "quality_score", "N/A")
        print(f"Quality Score    : {quality_score}")

    if result.get("dashboard_spec") is not None:
        dash = result["dashboard_spec"]
        visuals = dash.get("visuals", []) if isinstance(dash, dict) else getattr(dash, "visuals", [])
        print(f"Visual Count     : {len(visuals)}")

        if len(visuals) > 0:
            print("Visual Titles    :")
            for i, vis in enumerate(visuals, start=1):
                if isinstance(vis, dict):
                    title = vis.get("title", "N/A")
                    chart_type = vis.get("chart_type", "N/A")
                else:
                    title = getattr(vis, "title", "N/A")
                    chart_type = getattr(vis, "chart_type", "N/A")
                print(f"  {i}. {chart_type} - {title}")

    if len(errors) > 0:
        print("Errors:")
        for i, err in enumerate(errors, start=1):
            if isinstance(err, dict):
                stage = err.get("stage", "N/A")
                message = err.get("message", "N/A")
                recoverable = err.get("recoverable", "N/A")
            else:
                stage = getattr(err, "stage", "N/A")
                message = getattr(err, "message", "N/A")
                recoverable = getattr(err, "recoverable", "N/A")
            print(f"  {i}. Stage={stage} | Recoverable={recoverable} | Message={message}")


def print_pass():
    print("-" * 80)
    print("RESULT: PASS")
    print_separator()
    print()


def print_fail(error):
    print("-" * 80)
    print("RESULT: FAIL")
    print(f"ERROR : {error}")
    print_separator()
    print()


# ------------------------------------------------------------
# 3 NORMAL TEST CASES
# ------------------------------------------------------------

def normal_test_1():
    test_name = "Normal Test 1 - Successful workflow with main dataset"
    purpose = "Checks whether the complete workflow runs successfully on train.csv."

    try:
        file_path = "train.csv"

        wf = DashboardGeneratorWorkflow()
        result = wf.run(file_path)

        assert result["progress"]["current_stage"] == "completed"
        assert result["dataset_schema"] is not None
        assert result["quality_report"] is not None
        assert result["profile_report"] is not None
        assert result["analysis_result"] is not None
        assert result["dashboard_spec"] is not None

        print_test_header(test_name, purpose)
        print_result_summary(result)
        print_pass()

    except Exception as e:
        print_test_header(test_name, purpose)
        print_fail(e)


def normal_test_2():
    test_name = "Normal Test 2 - TWBX generation with main dataset"
    purpose = "Checks whether the workflow successfully generates TWBX output using train.csv."

    try:
        file_path = "train.csv"

        wf = DashboardGeneratorWorkflow()
        config = WorkflowConfig(output_format="twbx")
        result = wf.run(file_path, config=config)

        assert result["progress"]["current_stage"] == "completed"

        print_test_header(test_name, purpose)
        print_result_summary(result)
        print_pass()

    except Exception as e:
        print_test_header(test_name, purpose)
        print_fail(e)


def normal_test_3():
    test_name = "Normal Test 3 - Step-by-step execution with main dataset"
    purpose = "Checks whether all workflow stages run in the correct sequence using train.csv."

    try:
        file_path = "train.csv"
        visited_steps = []

        def on_progress(node_name, state):
            visited_steps.append(node_name)

        wf = DashboardGeneratorWorkflow()
        result = wf.run_step_by_step(file_path, on_progress=on_progress)

        expected_steps = ["validate", "profile", "analyze", "recommend", "generate", "finalize"]
        assert result["progress"]["current_stage"] == "completed"
        assert visited_steps == expected_steps

        print_test_header(test_name, purpose)
        print(f"Visited Steps    : {visited_steps}")
        print_result_summary(result)
        print_pass()

    except Exception as e:
        print_test_header(test_name, purpose)
        print_fail(e)


# ------------------------------------------------------------
# 1 EDGE TEST CASE
# ------------------------------------------------------------

def edge_test_3():
    test_name = "Edge Test 1 - Visualization limit constraint"
    purpose = "Checks whether the workflow respects the maximum visualization limit from config."

    try:
        file_path = "train.csv"

        wf = DashboardGeneratorWorkflow()
        config = WorkflowConfig(max_visualizations=1)
        result = wf.run(file_path, config=config)

        dash = result.get("dashboard_spec", {})
        visuals = dash.get("visuals", []) if isinstance(dash, dict) else getattr(dash, "visuals", [])

        assert result["progress"]["current_stage"] == "completed"
        assert len(visuals) <= 1

        print_test_header(test_name, purpose)
        print_result_summary(result)
        print_pass()

    except Exception as e:
        print_test_header(test_name, purpose)
        print_fail(e)


# ------------------------------------------------------------
# Run all tests
# ------------------------------------------------------------

if __name__ == "__main__":
    print_separator()
    print("COMPLETE WORKFLOW END-TO-END TESTING")
    print_separator()
    print()

    normal_test_1()
    normal_test_2()
    normal_test_3()
    edge_test_3()

    print_separator()
    print("ALL TESTS FINISHED")
    print_separator()