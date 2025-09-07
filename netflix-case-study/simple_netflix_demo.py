#!/usr/bin/env python3
"""
Simple Netflix ML Demo - Easy to Understand Version
==================================================
"""

def show_netflix_problem():
    print("\n" + "="*50)
    print("📺 NETFLIX'S BIG PROBLEM")
    print("="*50)
    print("Year: 2010")
    print("Problem: People were canceling Netflix subscriptions")
    print("Why? They couldn't find movies they liked!")
    print("\nExample:")
    print("• Sarah loves action movies")
    print("• Netflix shows her romantic comedies")
    print("• Sarah gets frustrated and cancels")
    print("\nResult: Netflix was losing customers fast! 💸")

def show_solution_idea():
    print("\n" + "="*50)
    print("💡 NETFLIX'S SMART IDEA")
    print("="*50)
    print("What if we could predict what people want to watch?")
    print("\nLike a smart friend who knows your taste:")
    print("• 'Hey Sarah, you loved Iron Man...'")
    print("• 'You should watch The Avengers!'")
    print("• 'People like you also enjoyed Thor!'")
    print("\nThis is called MACHINE LEARNING! 🤖")

def show_method1_simple():
    print("\n" + "="*50)
    print("🤝 METHOD 1: FIND SIMILAR PEOPLE")
    print("="*50)
    print("Idea: People with similar taste like similar movies")
    print("\nExample:")
    print("Sarah's ratings:")
    print("• Iron Man: ⭐⭐⭐⭐⭐")
    print("• Thor: ⭐⭐⭐⭐")
    print("• Titanic: ⭐⭐")
    print("\nJohn's ratings:")
    print("• Iron Man: ⭐⭐⭐⭐⭐")
    print("• Thor: ⭐⭐⭐⭐⭐")
    print("• Titanic: ⭐⭐")
    print("• The Avengers: ⭐⭐⭐⭐⭐")
    print("\n🎯 Netflix thinks: Sarah and John have similar taste!")
    print("💡 Recommendation: Show Sarah 'The Avengers'")

def show_method2_simple():
    print("\n" + "="*50)
    print("🎭 METHOD 2: FIND SIMILAR MOVIES")
    print("="*50)
    print("Idea: If you like one action movie, you might like others")
    print("\nExample:")
    print("Sarah watched and loved:")
    print("• Iron Man (Action, Superhero)")
    print("\nSimilar movies:")
    print("• The Avengers (Action, Superhero) ✅")
    print("• Batman (Action, Superhero) ✅")
    print("• Titanic (Romance, Drama) ❌")
    print("\n💡 Recommendation: Show Sarah more Action/Superhero movies")

def show_results():
    print("\n" + "="*50)
    print("📈 AMAZING RESULTS!")
    print("="*50)
    print("Before ML:")
    print("• 15 out of 100 people canceled every month 😞")
    print("• People watched for only 45 minutes")
    print("• People only found 20% of movies they liked")
    print("\nAfter ML:")
    print("• Only 5 out of 100 people cancel now! 😊")
    print("• People watch for 2.5 hours")
    print("• People find 80% more movies they like")
    print("\n💰 Netflix became super successful!")

def interactive_demo():
    print("\n" + "="*50)
    print("🎮 TRY IT YOURSELF!")
    print("="*50)
    
    print("Let's pretend you're Netflix's computer...")
    print("\nMeet Alice:")
    print("• Age: 25")
    print("• Loves: Sci-Fi movies")
    print("• Recently watched: Stranger Things ⭐⭐⭐⭐⭐")
    
    print("\nWhat should Netflix recommend to Alice?")
    print("1. Romantic comedy")
    print("2. Another Sci-Fi show")
    print("3. Documentary about cooking")
    
    choice = input("\nYour choice (1, 2, or 3): ")
    
    if choice == "2":
        print("\n🎉 CORRECT! You're thinking like Netflix's AI!")
        print("Alice loves Sci-Fi, so recommend more Sci-Fi!")
        print("Netflix would suggest: Black Mirror, The Matrix")
    else:
        print("\n🤔 Not quite! Alice loves Sci-Fi movies.")
        print("Netflix learned to recommend similar content.")
        print("Better choice: Another Sci-Fi show!")

def main():
    print("🎬" + "="*48)
    print("  NETFLIX'S SECRET: How They Keep You Watching")
    print("  (Simple Explanation for Beginners)")
    print("="*50)
    
    input("Press Enter to start...")
    show_netflix_problem()
    
    input("\nPress Enter to see Netflix's solution...")
    show_solution_idea()
    
    input("\nPress Enter to learn Method 1...")
    show_method1_simple()
    
    input("\nPress Enter to learn Method 2...")
    show_method2_simple()
    
    input("\nPress Enter to see the results...")
    show_results()
    
    input("\nPress Enter to try it yourself...")
    interactive_demo()
    
    print("\n" + "="*50)
    print("🎓 WHAT YOU LEARNED:")
    print("="*50)
    print("• Netflix had a big problem: people were leaving")
    print("• They used computers to learn what people like")
    print("• Method 1: Find people with similar taste")
    print("• Method 2: Find similar movies")
    print("• Result: People stay and watch more!")
    print("\nThis is how AI/Machine Learning helps businesses! 🚀")

if __name__ == "__main__":
    main()
