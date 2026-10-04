"""
app.py -- a simple command-line chat.

Usage:  python app.py
Type a question, or :debug to toggle showing retrieved chunks, or :quit to exit.
"""
from rag import answer


def main():
    print("Document chatbot. Ask about your docs (:debug to show chunks, :quit to exit).")
    debug = False
    while True:
        question = input("\nYou> ").strip()
        if not question:
            continue
        if question == ":quit":
            break
        if question == ":debug":
            debug = not debug
            print(f"Debug mode {'on' if debug else 'off'}.")
            continue

        result = answer(question)
        print(f"\nBot> {result['answer']}")

        if debug:
            print(f"\n  best similarity: {result['best_similarity']}"
                  f"   refused: {result['refused']} ({result['refused_at']})"
                  f"   tokens in/out: {result['input_tokens']}/{result['output_tokens']}")
            for s in result["sources"]:
                print(f"  {s['similarity']:.3f}  {s['id']}")
        elif not result["refused"]:
            files = sorted({s["source"] for s in result["sources"]})
            print(f"  (searched: {', '.join(files)})")


if __name__ == "__main__":
    main()
