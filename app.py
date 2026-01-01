import json
from graph import build_graph

if __name__ == "__main__":
    graph = build_graph()

    print("Type your request (or 'exit'):")
    while True:
        user_request = input("\nYou: ").strip()
        if user_request.lower() in ("exit", "quit"):
            break

        result = graph.invoke({"user_id": "prakash", "user_request": user_request})

        print("\n--- NOTES ---")
        for n in result.get("notes", []):
            print("-", n)

        print("\n--- WEEKLY SCHEDULE ---")
        print(json.dumps(result.get("weekly_schedule", {}), indent=2))
