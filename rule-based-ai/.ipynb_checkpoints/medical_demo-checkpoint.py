from rule_engine import Rule, RuleBasedAI

def medical_diagnosis_demo():
    ai = RuleBasedAI()
    
    # Define medical diagnosis rules (simplified)
    rules = [
        Rule(["fever", "cough"], "respiratory_infection", "R1: Fever + Cough"),
        Rule(["respiratory_infection", "chest_pain"], "pneumonia", "R2: Respiratory + Chest Pain"),
        Rule(["fever", "headache"], "viral_infection", "R3: Fever + Headache"),
        Rule(["nausea", "stomach_pain"], "gastritis", "R4: Nausea + Stomach Pain"),
        Rule(["pneumonia"], "prescribe_antibiotics", "R5: Pneumonia Treatment"),
        Rule(["viral_infection"], "prescribe_rest", "R6: Viral Treatment")
    ]
    
    for rule in rules:
        ai.add_rule(rule)
    
    # Patient symptoms
    symptoms = ["fever", "cough", "chest_pain"]
    print("MEDICAL DIAGNOSIS SYSTEM")
    print(f"Patient symptoms: {symptoms}")
    
    for symptom in symptoms:
        ai.add_fact(symptom)
    
    ai.infer()
    ai.explain()

if __name__ == "__main__":
    medical_diagnosis_demo()
