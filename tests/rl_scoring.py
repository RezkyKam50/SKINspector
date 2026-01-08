baseline = 1.0

factor = {
    "TECH_S": 0.25,
    "WRT_S": 0.10,
    "CLIN_RS": 0.20,
    "SFTY_S": 0.15,
    "CPLT_S": 0.10,
    "SPEC_S": 0.05,
    "EDU_S": 0.05,
    "LIM_S": 0.05,
    "STRC_S": 0.03,
    "ACT_S": 0.02
}

res = (
    baseline * factor["TECH_S"] +      # Core diagnostic accuracy
    baseline * factor["WRT_S"] +       # Professional communication
    baseline * factor["CLIN_RS"] +     # Differential diagnosis quality
    baseline * factor["SFTY_S"] +      # Critical for patient safety
    baseline * factor["CPLT_S"] +      # Thoroughness
    baseline * factor["SPEC_S"] +      # Diagnostic precision
    baseline * factor["EDU_S"] +       # Teaching value
    baseline * factor["LIM_S"] +       # Appropriate calibration
    baseline * factor["STRC_S"] +      # Organization
    baseline * factor["ACT_S"]         # Practical guidance
)