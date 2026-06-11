import argparse
import sys
from agent import config
from agent import loop


def main():
    parser = argparse.ArgumentParser(description="CLI Coding Agent (Bedrock)")
    parser.add_argument("--model", default=config.DEFAULT_MODEL,
                        choices=list(config.AVAILABLE_MODELS), help="Model alias to use")
    parser.add_argument("--region", default=config.REGION, help="AWS region")
    parser.add_argument("--profile", default=config.AWS_PROFILE, help="AWS profile name")
    parser.add_argument("--headless", action="store_true",
                        help="Run one task non-interactively (instruction from stdin or --task-file) and exit")
    parser.add_argument("--task-file", default=None,
                        help="Path to a file containing the task instruction (implies --headless)")
    args = parser.parse_args()

    config.REGION = args.region
    config.AWS_PROFILE = args.profile

    if args.headless or args.task_file:
        if args.task_file:
            with open(args.task_file) as f:
                task_text = f.read()
        else:
            task_text = sys.stdin.read()
        sys.exit(loop.run_headless(model_key=args.model, task_text=task_text))

    loop.run(model_key=args.model)


if __name__ == "__main__":
    main()
