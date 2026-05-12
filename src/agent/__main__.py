import argparse
from agent import config
from agent import loop


def main():
    parser = argparse.ArgumentParser(description="CLI Coding Agent (Bedrock)")
    parser.add_argument("--model", default=config.DEFAULT_MODEL,
                        choices=list(config.AVAILABLE_MODELS), help="Model alias to use")
    parser.add_argument("--region", default=config.REGION, help="AWS region")
    parser.add_argument("--profile", default=config.AWS_PROFILE, help="AWS profile name")
    args = parser.parse_args()

    config.REGION = args.region
    config.AWS_PROFILE = args.profile

    loop.run(model_key=args.model)


if __name__ == "__main__":
    main()
