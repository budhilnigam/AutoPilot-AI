"""Test Bedrock invoke with quotes in prompt to reproduce shlex issues."""
import asyncio
import traceback

from autopilot_ai.integrations.aws.bedrock import BedrockClient


async def main():
    client = BedrockClient()
    prompt = "Tell me a story that includes a single quote: Here's an example with 'unmatched quotes"
    try:
        print("Invoking with prompt containing quotes...")
        r = await client.invoke(prompt)
        print("SUCCESS:", r)
    except Exception:
        print("EXCEPTION:")
        traceback.print_exc()


if __name__ == '__main__':
    asyncio.run(main())
