"""Quick diagnostic: call aws_api('bedrock', 'list_foundation_models') and print results.

Run from the repo root with the project's venv active.
"""
import asyncio
import traceback

from autopilot_ai.core import config
from autopilot_ai.integrations.aws.tool import aws_api


async def main():
    print("Settings:")
    print("  aws_region:", config.settings.aws_region)
    print("  aws_profile:", config.settings.aws_profile)
    print("  has_access_keys:", bool(config.settings.aws_access_key_id and config.settings.aws_secret_access_key))
    print()
    try:
        print("Calling aws_api('bedrock','list_foundation_models', ...) ...")
        resp = await aws_api("bedrock", "list_foundation_models", {"byOutputModality": "TEXT"})
        print("Success. Response keys:", list(resp.keys()) if isinstance(resp, dict) else type(resp))
        print(resp)
    except Exception as e:
        print("Exception occurred:")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
