from rule_engine import Rule, RuleBasedAI

def explain_concepts():
    """
    Educational tutorial explaining Rule-Based AI concepts
    """
    print("🎓 RULE-BASED AI TUTORIAL")
    print("=" * 50)
    
    print("\n📚 WHAT IS RULE-BASED AI?")
    print("""
Rule-Based AI (also called Expert Systems) was the dominant AI approach from 1950-1980s.
Famous systems: MYCIN (medical diagnosis), DENDRAL (chemistry), XCON (computer configuration)

Key Concept: Human expertise is captured as IF-THEN rules
Example: "IF patient has fever AND cough THEN suspect respiratory infection"
""")
    
    print("\n🏗️  SYSTEM ARCHITECTURE:")
    print("""
1. KNOWLEDGE BASE: Collection of IF-THEN rules (expert knowledge)
2. WORKING MEMORY: Current facts about the situation
3. INFERENCE ENGINE: Matches rules against facts and draws conclusions
4. EXPLANATION FACILITY: Shows how conclusions were reached
""")
    
    print("\n⚙️  HOW INFERENCE WORKS (Forward Chaining):")
    print("""
1. Start with initial facts
2. Check all rules against current facts
3. If rule conditions match, add conclusion as new fact
4. Repeat until no new facts can be derived
5. This mimics how human experts reason step-by-step
""")

def simple_example():
    print("\n" + "="*50)
    print("🔬 SIMPLE EXAMPLE: Animal Classification")
    print("="*50)
    
    ai = RuleBasedAI()
    
    # Simple animal classification rules
    rules = [
        Rule(["has_fur", "gives_milk"], "mammal", "Mammal Rule"),
        Rule(["has_feathers", "lays_eggs"], "bird", "Bird Rule"),
        Rule(["mammal", "eats_meat"], "carnivore", "Carnivore Rule"),
        Rule(["mammal", "eats_plants"], "herbivore", "Herbivore Rule"),
        Rule(["carnivore", "hunts_in_packs"], "wolf", "Wolf Rule")
    ]
    
    print("\n📋 KNOWLEDGE BASE:")
    for i, rule in enumerate(rules, 1):
        print(f"   R{i}: {rule}")
        ai.add_rule(rule)
    
    print("\n🐺 CASE: Identifying an unknown animal")
    print("   Observations: has_fur, gives_milk, eats_meat, hunts_in_packs")
    
    # Add initial facts
    observations = ["has_fur", "gives_milk", "eats_meat", "hunts_in_packs"]
    for obs in observations:
        ai.add_fact(obs)
    
    print(f"\n💾 INITIAL WORKING MEMORY: {sorted(ai.facts)}")
    
    print("\n🔄 RUNNING INFERENCE ENGINE:")
    
    # Step-by-step inference
    iteration = 1
    while iteration <= 5:
        print(f"\n   Iteration {iteration}:")
        initial_facts = len(ai.facts)
        
        for rule in ai.rules:
            if rule.matches(ai.facts) and rule.action not in ai.facts:
                print(f"      ✅ {rule.name}: {rule.conditions} → {rule.action}")
                ai.facts.add(rule.action)
                ai.execution_log.append(f"Iteration {iteration}: {rule.name}")
        
        if len(ai.facts) == initial_facts:
            print("      ⏹️  No new facts derived. Stopping.")
            break
        
        print(f"      📊 Working memory: {sorted(ai.facts)}")
        iteration += 1
    
    print(f"\n🎯 FINAL CONCLUSION:")
    conclusions = [f for f in ai.facts if f not in observations]
    print(f"   From observations {observations}")
    print(f"   We concluded: {conclusions}")
    print(f"   The animal is a: WOLF")

def comparison_with_modern_ai():
    print("\n" + "="*50)
    print("🆚 RULE-BASED AI vs MODERN AI")
    print("="*50)
    
    print("""
RULE-BASED AI (1950s-1980s):
✅ Transparent reasoning (you can see why it decided)
✅ Expert knowledge directly encoded
✅ Deterministic and predictable
✅ Easy to debug and modify
❌ Brittle (fails on unexpected cases)
❌ Hard to handle uncertainty
❌ Requires manual knowledge encoding

MODERN AI (Neural Networks, LLMs):
✅ Learns from data automatically
✅ Handles uncertainty and ambiguity
✅ Works on complex, unstructured problems
❌ "Black box" - hard to explain decisions
❌ Requires lots of data
❌ Can be unpredictable
""")

if __name__ == "__main__":
    explain_concepts()
    simple_example()
    comparison_with_modern_ai()
    
    print("\n" + "="*50)
    print("🚀 Ready to try the medical demo? Run: python medical_demo.py")
    print("🌐 Want to see it visually? Open: visualizer.html")
    print("="*50)
