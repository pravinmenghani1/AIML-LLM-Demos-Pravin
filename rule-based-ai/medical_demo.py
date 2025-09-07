from rule_engine import Rule, RuleBasedAI
import time

def detailed_medical_demo():
    print("=" * 60)
    print("🏥 RULE-BASED AI MEDICAL DIAGNOSIS SYSTEM")
    print("   (Classic Expert System - 1950s-1980s Style)")
    print("=" * 60)
    
    # Step 1: Create the AI system
    print("\n📋 STEP 1: Creating the Expert System")
    print("   - Initializing rule engine...")
    print("   - Setting up knowledge base...")
    ai = RuleBasedAI()
    
    # Step 2: Define the knowledge base (rules)
    print("\n🧠 STEP 2: Loading Medical Knowledge Base")
    print("   Adding IF-THEN rules (like doctors' decision trees):")
    
    rules = [
        Rule(["fever", "cough"], "respiratory_infection", "Respiratory Rule"),
        Rule(["respiratory_infection", "chest_pain"], "pneumonia", "Pneumonia Rule"),
        Rule(["fever", "headache", "body_aches"], "flu", "Flu Rule"),
        Rule(["nausea", "stomach_pain"], "gastritis", "Gastritis Rule"),
        Rule(["pneumonia"], "prescribe_antibiotics", "Antibiotic Treatment"),
        Rule(["flu"], "prescribe_rest_fluids", "Flu Treatment"),
        Rule(["gastritis"], "prescribe_antacids", "Gastritis Treatment"),
        Rule(["fever", "rash"], "viral_infection", "Viral Rule"),
        Rule(["chest_pain", "shortness_of_breath"], "cardiac_concern", "Cardiac Rule")
    ]
    
    for i, rule in enumerate(rules, 1):
        print(f"   R{i}: IF {' AND '.join(rule.conditions)} THEN {rule.action}")
        ai.add_rule(rule)
    
    # Step 3: Present patient case
    print("\n🤒 STEP 3: Patient Case Presentation")
    print("   A patient walks into the clinic with these symptoms:")
    
    # Let's try different scenarios
    scenarios = [
        {
            "name": "Patient A - Respiratory Case",
            "symptoms": ["fever", "cough", "chest_pain"],
            "description": "Adult with high fever, persistent cough, and chest pain"
        },
        {
            "name": "Patient B - Flu Case", 
            "symptoms": ["fever", "headache", "body_aches"],
            "description": "Young adult with fever, headache, and body aches"
        }
    ]
    
    for scenario in scenarios:
        print(f"\n" + "="*50)
        print(f"🔍 ANALYZING: {scenario['name']}")
        print(f"   Description: {scenario['description']}")
        print(f"   Reported symptoms: {scenario['symptoms']}")
        
        # Reset AI for new patient
        ai.facts.clear()
        ai.execution_log.clear()
        
        # Step 4: Add symptoms to working memory
        print(f"\n💾 STEP 4: Adding symptoms to working memory...")
        for symptom in scenario['symptoms']:
            ai.add_fact(symptom)
            print(f"   ✓ Added: {symptom}")
        
        print(f"\n   Initial working memory: {sorted(ai.facts)}")
        
        # Step 5: Run inference engine
        print(f"\n⚙️  STEP 5: Running Inference Engine...")
        print("   (Forward-chaining: checking rules against facts)")
        
        # Manual step-by-step inference for demonstration
        iteration = 1
        while iteration <= 5:
            print(f"\n   🔄 Iteration {iteration}:")
            fired_any = False
            
            for rule in ai.rules:
                if rule.matches(ai.facts) and rule.action not in ai.facts:
                    print(f"      🔥 RULE FIRED: {rule.name}")
                    print(f"         Conditions met: {rule.conditions}")
                    print(f"         New conclusion: {rule.action}")
                    ai.facts.add(rule.action)
                    ai.execution_log.append(f"Iteration {iteration}: {rule.name} → {rule.action}")
                    fired_any = True
                    time.sleep(0.5)  # Pause for dramatic effect
            
            if not fired_any:
                print(f"      ⏹️  No more rules can fire. Inference complete.")
                break
            
            print(f"      📊 Working memory now: {sorted(ai.facts)}")
            iteration += 1
        
        # Step 6: Present diagnosis and treatment
        print(f"\n📋 STEP 6: FINAL DIAGNOSIS & TREATMENT")
        
        # Separate symptoms from diagnoses and treatments
        symptoms = [f for f in ai.facts if f in scenario['symptoms']]
        diagnoses = [f for f in ai.facts if f not in scenario['symptoms'] and not f.startswith('prescribe_')]
        treatments = [f for f in ai.facts if f.startswith('prescribe_')]
        
        print(f"   Symptoms observed: {symptoms}")
        print(f"   Diagnoses reached: {diagnoses}")
        print(f"   Treatments recommended: {treatments}")
        
        print(f"\n📈 REASONING CHAIN:")
        for log in ai.execution_log:
            print(f"   → {log}")

def interactive_demo():
    print("\n" + "="*60)
    print("🎮 INTERACTIVE MODE - Build Your Own Case!")
    print("="*60)
    
    ai = RuleBasedAI()
    
    # Add rules
    rules = [
        Rule(["fever", "cough"], "respiratory_infection", "Respiratory Rule"),
        Rule(["respiratory_infection", "chest_pain"], "pneumonia", "Pneumonia Rule"),
        Rule(["fever", "headache", "body_aches"], "flu", "Flu Rule"),
        Rule(["pneumonia"], "prescribe_antibiotics", "Antibiotic Treatment"),
        Rule(["flu"], "prescribe_rest_fluids", "Flu Treatment")
    ]
    
    for rule in rules:
        ai.add_rule(rule)
    
    print("\nAvailable symptoms: fever, cough, chest_pain, headache, body_aches")
    print("Type symptoms one by one (press Enter after each, 'done' to finish):")
    
    while True:
        symptom = input("Add symptom: ").strip().lower()
        if symptom == 'done':
            break
        if symptom:
            ai.add_fact(symptom)
            print(f"✓ Added: {symptom}")
    
    print(f"\nRunning diagnosis on: {sorted(ai.facts)}")
    ai.infer()
    ai.explain()

if __name__ == "__main__":
    detailed_medical_demo()
    
    print("\n" + "="*60)
    choice = input("Want to try interactive mode? (y/n): ").lower()
    if choice == 'y':
        interactive_demo()
