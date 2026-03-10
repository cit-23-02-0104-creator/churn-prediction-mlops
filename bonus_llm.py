import time

def llm_retention_generator(tenure, service, monthly_charges):
    """
    Demonstrating LLM-Based Retention Incentive Generator 
    (Simulating Prompt Engineering & Output Formatting)
    """
    
    
    system_prompt = "You are a Customer Retention Expert."
    user_prompt = f"""
    Customer Data: Tenure: {tenure} months, Service: {service}, Bill: ${monthly_charges}
    Task: If churn is predicted, generate a personalized incentive.
    Constraint: Offer 15% discount and free upgrade.
    """
    
    print(f"[PROMPT SENT TO LLM]:\n{system_prompt}\n{user_prompt}")
    print("\nProcessing with LLM...")
    time.sleep(1) 

    
    incentive_message = f"""
    --------------------------------------------------
    GENERATED RETENTION INCENTIVE (LLM OUTPUT)
    --------------------------------------------------
    Dear Valued Customer,
    
    We noticed you've been with us for {tenure} months as a {service} user. 
    To show our appreciation, we are offering you a special 15% loyalty 
    discount on your monthly bill of ${monthly_charges} for the next 6 months. 
    
    Plus, we are upgrading your {service} speed for FREE! 
    We value your loyalty and look forward to serving you further.
    
    Best Regards,
    Customer Success Team
    --------------------------------------------------
    """
    return incentive_message

# Run the demonstration
if __name__ == "__main__":
    output = llm_retention_generator(36, "Fiber optic", 75.50)
    print(output)