class Rule:
    def __init__(self, conditions, action, name=""):
        self.conditions = conditions  # List of facts that must be true
        self.action = action          # New fact to add if conditions met
        self.name = name             # Human-readable rule name
    
    def matches(self, facts):
        """Check if all conditions are satisfied by current facts"""
        return all(fact in facts for fact in self.conditions)
    
    def __str__(self):
        return f"IF {' AND '.join(self.conditions)} THEN {self.action}"

class RuleBasedAI:
    """
    Classic Expert System Implementation (1950s-1980s style)
    
    How it works:
    1. Facts are stored in working memory
    2. Rules are checked against current facts
    3. When rule conditions match, new facts are derived
    4. Process repeats until no new facts can be derived
    """
    
    def __init__(self):
        self.rules = []           # Knowledge base (IF-THEN rules)
        self.facts = set()        # Working memory (current facts)
        self.execution_log = []   # Trace of rule firings
    
    def add_rule(self, rule):
        """Add a rule to the knowledge base"""
        self.rules.append(rule)
    
    def add_fact(self, fact):
        """Add a fact to working memory"""
        self.facts.add(fact)
    
    def infer(self, max_iterations=10):
        """
        Forward-chaining inference engine
        Repeatedly applies rules until no new facts can be derived
        """
        for iteration in range(max_iterations):
            fired = False
            for rule in self.rules:
                # Check if rule can fire (conditions met, conclusion not already known)
                if rule.matches(self.facts) and rule.action not in self.facts:
                    self.facts.add(rule.action)
                    self.execution_log.append(f"Rule '{rule.name}' fired: {rule.action}")
                    fired = True
            
            # Stop if no rules fired in this iteration
            if not fired:
                break
        
        return self.facts
    
    def explain(self):
        """Provide detailed explanation of reasoning process"""
        print("=== RULE-BASED AI EXECUTION TRACE ===")
        print(f"Final facts derived: {sorted(self.facts)}")
        print("\nStep-by-step reasoning:")
        for i, log in enumerate(self.execution_log, 1):
            print(f"  {i}. {log}")
        
        if not self.execution_log:
            print("  No rules were fired (no new conclusions drawn)")
    
    def show_knowledge_base(self):
        """Display all rules in the knowledge base"""
        print("=== KNOWLEDGE BASE (Rules) ===")
        for i, rule in enumerate(self.rules, 1):
            print(f"R{i}: {rule}")
    
    def show_working_memory(self):
        """Display current facts in working memory"""
        print(f"=== WORKING MEMORY (Facts) ===")
        print(f"Current facts: {sorted(self.facts)}")
