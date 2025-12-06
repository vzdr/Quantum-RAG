"""
Medical Diagnosis Synthetic Data Generator

Generates 60 realistic medical document chunks for 6 diseases with strategic
symptom overlap to demonstrate QUBO-RAG's diversity advantage.

Design: Overlapping symptoms create retrieval challenges where Naive RAG
retrieves redundant docs, while QUBO RAG retrieves diverse disease perspectives.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple


# Disease definitions with overlapping and unique features
DISEASES = {
    'lupus': {
        'full_name': 'Systemic Lupus Erythematosus (SLE)',
        'overlapping_symptoms': ['chronic fatigue', 'joint pain', 'fever', 'skin rash'],
        'unique_features': ['butterfly-shaped facial rash', 'photosensitivity', 'kidney involvement', 'antinuclear antibodies (ANA)'],
        'diagnostic_tests': ['ANA test', 'anti-dsDNA antibodies', 'complement levels', 'urinalysis'],
        'treatments': ['hydroxychloroquine', 'corticosteroids', 'immunosuppressants', 'NSAIDs'],
        'risk_factors': ['female gender', 'age 15-45', 'genetics', 'environmental triggers'],
        'complications': ['kidney disease', 'cardiovascular disease', 'neurological symptoms', 'infections']
    },
    'rheumatoid_arthritis': {
        'full_name': 'Rheumatoid Arthritis (RA)',
        'overlapping_symptoms': ['joint pain', 'morning stiffness', 'fatigue', 'fever'],
        'unique_features': ['symmetric joint involvement', 'rheumatoid nodules', 'hand/wrist deformities', 'positive RF/anti-CCP'],
        'diagnostic_tests': ['rheumatoid factor (RF)', 'anti-CCP antibodies', 'ESR/CRP', 'joint X-rays'],
        'treatments': ['methotrexate', 'biologics (TNF inhibitors)', 'corticosteroids', 'physical therapy'],
        'risk_factors': ['family history', 'smoking', 'female gender', 'age 40-60'],
        'complications': ['joint damage', 'osteoporosis', 'carpal tunnel syndrome', 'lung disease']
    },
    'lyme_disease': {
        'full_name': 'Lyme Disease',
        'overlapping_symptoms': ['fatigue', 'fever', 'joint pain', 'skin rash'],
        'unique_features': ['erythema migrans (bull\'s-eye rash)', 'tick exposure', 'facial palsy', 'Lyme carditis'],
        'diagnostic_tests': ['ELISA and Western blot', 'clinical diagnosis', 'tick exposure history', 'erythema migrans observation'],
        'treatments': ['doxycycline', 'amoxicillin', 'ceftriaxone (IV for severe cases)', 'symptom management'],
        'risk_factors': ['outdoor activities', 'tick-endemic areas', 'spring/summer months', 'lack of tick prevention'],
        'complications': ['Lyme arthritis', 'neurological symptoms', 'cardiac conduction abnormalities', 'post-treatment syndrome']
    },
    'fibromyalgia': {
        'full_name': 'Fibromyalgia',
        'overlapping_symptoms': ['chronic fatigue', 'widespread pain', 'sleep disturbances', 'cognitive difficulties'],
        'unique_features': ['tender points', 'no inflammation markers', 'pain amplification', 'absence of joint damage'],
        'diagnostic_tests': ['clinical criteria (widespread pain index)', 'symptom severity scale', 'exclusion of other conditions', 'tender point examination'],
        'treatments': ['pregabalin', 'duloxetine', 'cognitive behavioral therapy', 'exercise programs'],
        'risk_factors': ['female gender', 'family history', 'stress/trauma', 'age 20-50'],
        'complications': ['quality of life impairment', 'depression/anxiety', 'social isolation', 'work disability']
    },
    'chronic_fatigue': {
        'full_name': 'Chronic Fatigue Syndrome (ME/CFS)',
        'overlapping_symptoms': ['severe fatigue', 'cognitive impairment', 'sleep problems', 'muscle pain'],
        'unique_features': ['post-exertional malaise (PEM)', 'unrefreshing sleep', 'orthostatic intolerance', '6+ months duration'],
        'diagnostic_tests': ['clinical criteria (IOM 2015)', 'exclusion of other conditions', 'tilt table test', 'symptom diary'],
        'treatments': ['pacing/energy management', 'cognitive behavioral therapy', 'graded exercise (controversial)', 'symptom-targeted medications'],
        'risk_factors': ['viral infections', 'immune dysregulation', 'stress', 'genetics'],
        'complications': ['severe disability', 'loss of employment', 'social isolation', 'depression']
    },
    'hypothyroidism': {
        'full_name': 'Hypothyroidism',
        'overlapping_symptoms': ['fatigue', 'weight gain', 'joint pain', 'depression'],
        'unique_features': ['cold intolerance', 'dry skin', 'hair loss', 'elevated TSH'],
        'diagnostic_tests': ['TSH level', 'free T4', 'thyroid antibodies (TPO)', 'thyroid ultrasound'],
        'treatments': ['levothyroxine replacement', 'dose titration', 'regular TSH monitoring', 'lifelong therapy'],
        'risk_factors': ['autoimmune disease', 'iodine deficiency', 'female gender', 'age >60'],
        'complications': ['myxedema coma', 'heart disease', 'infertility', 'peripheral neuropathy']
    }
}


def create_symptom_document(disease: str, disease_data: Dict, chunk_num: int) -> str:
    """Generate a symptom description document with overlap."""
    full_name = disease_data['full_name']
    overlapping = disease_data['overlapping_symptoms']
    unique = disease_data['unique_features']

    if chunk_num == 1:
        # Focus on overlapping symptoms (creates retrieval challenge)
        content = (
            f"{full_name} patients commonly present with {overlapping[0]}, {overlapping[1]}, "
            f"and {overlapping[2]}. These symptoms can persist for weeks to months and significantly "
            f"impact daily functioning. Many patients also experience {overlapping[3]}, which can "
            f"be mistaken for other conditions. Early recognition of symptom patterns is crucial "
            f"for timely diagnosis and treatment initiation."
        )
    else:
        # Include unique features (helps differentiation)
        content = (
            f"Distinguishing features of {full_name} include {unique[0]} and {unique[1]}. "
            f"Unlike some similar conditions, patients often exhibit {unique[2]}. "
            f"The presence of {unique[3]} is particularly characteristic and aids in "
            f"differential diagnosis. Clinical examination should specifically assess for "
            f"these unique manifestations to distinguish from other conditions presenting with {overlapping[0]}."
        )

    return content


def create_diagnostic_document(disease: str, disease_data: Dict, chunk_num: int) -> str:
    """Generate diagnostic criteria and testing information."""
    full_name = disease_data['full_name']
    tests = disease_data['diagnostic_tests']
    unique = disease_data['unique_features']

    if chunk_num == 1:
        content = (
            f"Diagnosis of {full_name} requires {tests[0]} as a primary screening tool. "
            f"Confirmatory testing includes {tests[1]} and {tests[2]}, which help establish "
            f"the diagnosis with greater certainty. Clinical criteria emphasize the presence of "
            f"{unique[0]}, though not all patients present with classic findings. "
            f"A systematic diagnostic approach combining laboratory and clinical assessment is essential."
        )
    else:
        content = (
            f"Additional diagnostic workup for {full_name} involves {tests[3]} to assess "
            f"disease severity and organ involvement. Serial testing may be necessary to track "
            f"disease progression. Differential diagnosis should consider conditions with similar "
            f"presentations but distinct {tests[1]} results. Clinical judgment remains important "
            f"as laboratory findings must be interpreted in the clinical context of symptom presentation."
        )

    return content


def create_risk_factors_document(disease: str, disease_data: Dict, chunk_num: int) -> str:
    """Generate risk factors and epidemiology information."""
    full_name = disease_data['full_name']
    risks = disease_data['risk_factors']

    if chunk_num == 1:
        content = (
            f"{full_name} occurs more frequently in individuals with {risks[0]} and {risks[1]}. "
            f"Epidemiological studies show peak incidence during {risks[2]}, though cases can occur "
            f"outside this typical demographic. Understanding risk factors helps identify at-risk "
            f"populations for early screening and intervention. Environmental and genetic factors "
            f"interact to influence disease susceptibility and severity."
        )
    else:
        content = (
            f"Genetic predisposition plays a significant role in {full_name}, particularly {risks[3]}. "
            f"Patients with {risks[0]} should be monitored more closely for early disease manifestations. "
            f"Modifiable risk factors, where they exist, represent important targets for prevention "
            f"strategies. Population-based screening may be warranted for high-risk groups with multiple "
            f"predisposing factors."
        )

    return content


def create_treatment_document(disease: str, disease_data: Dict, chunk_num: int) -> str:
    """Generate treatment approach information."""
    full_name = disease_data['full_name']
    treatments = disease_data['treatments']

    if chunk_num == 1:
        content = (
            f"First-line treatment for {full_name} typically involves {treatments[0]}, "
            f"which has demonstrated efficacy in controlled trials. Many patients require "
            f"additional therapy with {treatments[1]} for optimal symptom control. "
            f"Treatment response should be monitored regularly with clinical assessment and "
            f"appropriate laboratory testing. Dose adjustments may be necessary based on "
            f"disease activity and patient tolerance."
        )
    else:
        content = (
            f"For refractory {full_name}, second-line options include {treatments[2]} and {treatments[3]}. "
            f"Combination therapy may provide superior outcomes compared to monotherapy in severe cases. "
            f"Treatment goals focus on symptom relief, preventing complications, and maintaining quality "
            f"of life. Regular follow-up is essential to assess therapeutic efficacy and adjust management "
            f"strategies as needed based on individual patient response."
        )

    return content


def create_complications_document(disease: str, disease_data: Dict, chunk_num: int) -> str:
    """Generate complications and prognosis information."""
    full_name = disease_data['full_name']
    complications = disease_data['complications']

    if chunk_num == 1:
        content = (
            f"Untreated or poorly controlled {full_name} can lead to {complications[0]}, "
            f"which significantly impacts long-term prognosis. Patients may also develop "
            f"{complications[1]}, requiring specialized management and monitoring. "
            f"Early diagnosis and appropriate treatment can prevent or minimize these complications. "
            f"Regular screening for complications is an essential component of comprehensive disease management."
        )
    else:
        content = (
            f"Additional complications of {full_name} include {complications[2]} and {complications[3]}. "
            f"Prognosis varies based on disease severity, treatment response, and development of complications. "
            f"Multidisciplinary care may be necessary to address the diverse manifestations and complications. "
            f"Patient education about warning signs of complications enables earlier intervention and "
            f"improved outcomes through prompt medical attention."
        )

    return content


def generate_all_documents() -> List[Tuple[str, str]]:
    """
    Generate all 60 medical documents (6 diseases × 10 chunks).

    Returns:
        List of (filename, content) tuples
    """
    documents = []

    for disease, disease_data in DISEASES.items():
        # 2 symptom documents
        for i in range(1, 3):
            filename = f"{disease}_symptoms_{i}.txt"
            content = create_symptom_document(disease, disease_data, i)
            documents.append((filename, content))

        # 2 diagnostic documents
        for i in range(1, 3):
            filename = f"{disease}_diagnosis_{i}.txt"
            content = create_diagnostic_document(disease, disease_data, i)
            documents.append((filename, content))

        # 2 risk factor documents
        for i in range(1, 3):
            filename = f"{disease}_risk_factors_{i}.txt"
            content = create_risk_factors_document(disease, disease_data, i)
            documents.append((filename, content))

        # 2 treatment documents
        for i in range(1, 3):
            filename = f"{disease}_treatment_{i}.txt"
            content = create_treatment_document(disease, disease_data, i)
            documents.append((filename, content))

        # 2 complications documents
        for i in range(1, 3):
            filename = f"{disease}_complications_{i}.txt"
            content = create_complications_document(disease, disease_data, i)
            documents.append((filename, content))

    return documents


def save_documents(output_dir: str, documents: List[Tuple[str, str]]):
    """Save all documents to output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for filename, content in documents:
        filepath = output_path / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

    print(f"[OK] Generated {len(documents)} medical documents in {output_dir}")

    # Print statistics
    diseases = set(filename.split('_')[0] for filename, _ in documents)
    print(f"[OK] Diseases: {', '.join(sorted(diseases))}")
    print(f"[OK] Documents per disease: {len(documents) // len(diseases)}")

    # Show sample
    print(f"\nSample document ({documents[0][0]}):")
    print(f"Length: {len(documents[0][1])} chars")
    print(f"Content: {documents[0][1][:200]}...")


def main():
    """Generate and save all medical diagnosis documents."""
    print("Generating Medical Diagnosis Synthetic Dataset...")
    print("=" * 70)

    # Get script directory
    script_dir = Path(__file__).parent

    # Generate documents
    documents = generate_all_documents()

    # Save to current directory
    save_documents(script_dir, documents)

    print("\n" + "=" * 70)
    print("[OK] Data generation complete!")
    print("\nNext steps:")
    print("1. Run RAG_System.ipynb to index the medical documents")
    print("2. Test queries showcasing QUBO diversity advantage")
    print("3. Compare Naive vs MMR vs QUBO retrieval results")


if __name__ == '__main__':
    main()
