"""
Generate Strategic Synthetic Medical Data for QUBO-RAG Hard Case Demonstrations

This module creates medical diagnosis datasets with controlled similarity structure to
showcase QUBO-RAG's competitive advantages over MMR (Maximal Marginal Relevance).

Key Design Principle: "Similarity Cluster Traps"
- MODERATE intra-cluster similarity (0.65-0.80), not too high (0.9+)
- Multiple high-relevance clusters (all scoring 0.60+ on query similarity)
- Lower inter-cluster similarity (0.30-0.50)
- This creates scenarios where MMR's greedy selection gets trapped in local optima

Why This Exposes MMR's Weakness:
- MMR picks first high-scoring document (likely from Cluster A)
- Then it penalizes documents similar to first selection
- BUT other documents in Cluster A still score well on MMR formula
- Result: MMR picks 3-4 documents from same cluster, missing diverse perspectives

Why QUBO Wins:
- Global optimization considers ALL pairwise similarities simultaneously
- Penalizes SUM of similarities across entire selection (not just MAX)
- Finds optimal set: balanced across clusters for maximum information coverage
"""

import os
from pathlib import Path
from typing import List, Tuple, Dict
import random


# ============================================================================
# SCENARIO A: Symptom Overlap Trap
# ============================================================================

def generate_symptom_overlap_scenario() -> Tuple[List[Tuple[str, str]], Dict[str, any]]:
    """
    Generate hard-case scenario with overlapping symptoms across 4 diseases.

    Query: "Patient presents with chronic fatigue, joint pain, and low-grade fever"

    Cluster Design:
    - Chronic Fatigue Syndrome: High fatigue focus (query sim ~0.65-0.70)
    - Rheumatoid Arthritis: High joint pain focus (query sim ~0.68-0.72)
    - Lupus: Balanced all symptoms (query sim ~0.72-0.75) - THE TRAP!
    - Fibromyalgia: Widespread pain + fatigue (query sim ~0.62-0.66)

    Returns:
        List of (filename, content) tuples + ground truth metadata
    """
    documents = []

    # ----------------------------------------------------
    # CLUSTER 1: Chronic Fatigue Syndrome (8 documents)
    # Target intra-cluster similarity: 0.70
    # ----------------------------------------------------
    cfs_templates = [
        # Symptoms document
        "Chronic fatigue syndrome primarily manifests with profound, persistent exhaustion that doesn't improve with rest. Patients experience severe tiredness lasting six months or more, often accompanied by cognitive difficulties, muscle aches, and joint discomfort without swelling. Some individuals report low-grade fever and tender lymph nodes. The fatigue is so overwhelming it significantly reduces daily activity levels.",

        # Diagnosis document
        "Diagnosing chronic fatigue syndrome requires excluding other conditions that cause similar symptoms. Physicians look for persistent fatigue lasting at least 6 months, along with four or more additional symptoms including unrefreshing sleep, post-exertional malaise, memory problems, or muscle pain. Laboratory tests help rule out thyroid disorders, sleep apnea, and depression. No specific diagnostic test confirms CFS.",

        # Risk factors document
        "Chronic fatigue syndrome affects women more frequently than men, typically developing between ages 40-60. Risk factors include viral infections, immune system dysregulation, and hormonal imbalances. Stress and psychological factors may contribute but aren't primary causes. Family history sometimes suggests genetic predisposition. The exact cause remains unknown.",

        # Treatment document
        "Managing chronic fatigue syndrome focuses on symptom relief since no cure exists. Treatment includes paced activity management to avoid exacerbating fatigue, cognitive behavioral therapy for coping strategies, and medications for pain or sleep disturbances. Some patients benefit from graded exercise therapy. Antidepressants may help with sleep and pain management.",

        # Complications document
        "Chronic fatigue syndrome can lead to prolonged disability and social isolation. Patients often struggle with maintaining employment due to severe exhaustion and cognitive difficulties. Depression and anxiety frequently develop as secondary conditions. Quality of life significantly decreases. Some experience improvement over time, but many have persistent symptoms for years.",

        # Pathophysiology document
        "The underlying mechanisms of chronic fatigue syndrome involve complex interactions between immune, neurological, and endocrine systems. Research suggests chronic immune activation, mitochondrial dysfunction, and autonomic nervous system abnormalities. Brain imaging shows altered activity in regions controlling fatigue perception. Cellular energy metabolism may be impaired.",

        # Prognosis document
        "Chronic fatigue syndrome prognosis varies widely among patients. Some experience gradual improvement over several years, while others have persistent symptoms for decades. Younger patients and those with less severe initial symptoms tend to have better outcomes. Complete recovery is uncommon but possible. Most learn to manage symptoms and adapt activities.",

        # Clinical presentation variant
        "Patients with chronic fatigue syndrome typically present after months of unexplained exhaustion that profoundly impacts their functioning. The fatigue is often accompanied by muscle soreness, difficulty concentrating, headaches, and unrefreshing sleep. Many report feeling drained after minimal physical or mental exertion. Joint pain without inflammation is common.",
    ]

    for i, content in enumerate(cfs_templates):
        documents.append((f"cfs_doc_{i+1}.txt", content))

    # ----------------------------------------------------
    # CLUSTER 2: Rheumatoid Arthritis (8 documents)
    # Target intra-cluster similarity: 0.72
    # ----------------------------------------------------
    ra_templates = [
        # Symptoms document
        "Rheumatoid arthritis causes painful, swollen joints primarily affecting hands, wrists, and knees. Patients experience morning stiffness lasting over an hour, with joints feeling warm and tender. Symmetrical joint involvement is characteristic. Many patients report fatigue, low-grade fever, and general malaise. Joint deformities may develop without treatment.",

        # Diagnosis document
        "Diagnosing rheumatoid arthritis involves clinical examination, blood tests, and imaging. Rheumatoid factor and anti-CCP antibodies are often elevated. Inflammatory markers like ESR and CRP are typically high during active disease. X-rays or ultrasound reveal joint inflammation and erosions. Diagnosis requires persistent symptoms for at least 6 weeks.",

        # Risk factors document
        "Rheumatoid arthritis has strong genetic components, with certain HLA genes increasing risk substantially. Women are affected three times more than men, typically developing disease between ages 30-50. Smoking significantly increases risk and severity. Hormonal factors may play a role. Family history is an important risk factor.",

        # Treatment document
        "Treating rheumatoid arthritis requires disease-modifying antirheumatic drugs (DMARDs) like methotrexate to prevent joint damage. Biologic agents targeting specific immune pathways are used for severe cases. NSAIDs and corticosteroids help control pain and inflammation. Physical therapy maintains joint function. Early aggressive treatment prevents deformities.",

        # Complications document
        "Untreated rheumatoid arthritis leads to progressive joint destruction, deformities, and disability. Patients may develop cardiovascular disease, lung inflammation, and osteoporosis. Chronic inflammation increases infection risk. Work disability is common in severe cases. Proper treatment significantly reduces complication rates.",

        # Pathophysiology document
        "Rheumatoid arthritis is an autoimmune disease where the immune system attacks joint linings, causing synovial inflammation. This triggers production of inflammatory cytokines that damage cartilage and bone. Immune cells infiltrate joints, forming pannus tissue that erodes structures. The systemic inflammation affects multiple organs beyond joints.",

        # Prognosis document
        "With modern treatment, rheumatoid arthritis prognosis has improved dramatically. Early intervention with DMARDs can achieve remission in many patients. However, disease often fluctuates with periods of flares and remission. Joint damage that occurs early may be irreversible. Ongoing monitoring and treatment adjustment optimize outcomes.",

        # Clinical presentation variant
        "Rheumatoid arthritis typically presents with gradual onset of joint pain and swelling, particularly in small joints of hands and feet. Morning stiffness is pronounced, often lasting several hours. Patients may experience fatigue, mild fever, and muscle aches. Joint symptoms are usually symmetric. Tender, boggy joint swelling is characteristic on examination.",
    ]

    for i, content in enumerate(ra_templates):
        documents.append((f"ra_doc_{i+1}.txt", content))

    # ----------------------------------------------------
    # CLUSTER 3: Lupus (8 documents) - THE TRAP!
    # Balanced symptoms: fatigue + joint pain + fever
    # Target intra-cluster similarity: 0.68
    # Highest query similarity: ~0.72-0.75
    # ----------------------------------------------------
    lupus_templates = [
        # Symptoms document
        "Systemic lupus erythematosus presents with diverse symptoms including profound fatigue, joint pain and swelling, and recurrent low-grade fevers. The characteristic butterfly rash across cheeks and nose appears in many patients. Photosensitivity, mouth ulcers, and hair loss are common. Kidney involvement may cause protein in urine. Symptoms fluctuate with flares and remissions.",

        # Diagnosis document
        "Diagnosing lupus requires meeting clinical and laboratory criteria. Antinuclear antibodies (ANA) are positive in over 95% of patients. Specific antibodies like anti-dsDNA and anti-Smith support diagnosis. Complete blood counts may show anemia, low white cells, or low platelets. Complement levels often decrease during active disease. Multiple organ involvement is typical.",

        # Risk factors document
        "Lupus predominantly affects women of childbearing age, with a 9:1 female-to-male ratio. African American, Hispanic, and Asian women have higher risk than Caucasians. Genetic factors play a role, with family history increasing susceptibility. Hormonal factors likely contribute to female predominance. Environmental triggers like sunlight and certain medications can precipitate disease.",

        # Treatment document
        "Lupus treatment depends on disease severity and organ involvement. Antimalarials like hydroxychloroquine form the backbone for most patients. Corticosteroids control inflammation during flares. Immunosuppressants such as azathioprine or mycophenolate treat severe manifestations. Belimumab, a biologic agent, helps some patients. Sun protection and lifestyle modifications are essential.",

        # Complications document
        "Lupus complications arise from both disease activity and treatment. Kidney failure can develop from lupus nephritis. Cardiovascular disease risk increases significantly. Central nervous system involvement may cause seizures or psychosis. Increased infection risk results from immunosuppression. Pregnancy carries higher risks. Joint pain is common but usually non-erosive.",

        # Pathophysiology document
        "Systemic lupus erythematosus is a systemic autoimmune disease characterized by loss of tolerance to self-antigens. Autoantibodies form immune complexes that deposit in multiple organs, triggering inflammation. Type I interferon plays a central role in disease pathogenesis. Defective clearance of apoptotic cells and neutrophil extracellular traps contribute. Genetic and environmental factors interact.",

        # Prognosis document
        "Lupus prognosis has improved markedly with modern treatment. Five-year survival exceeds 95% overall. However, disease severity varies enormously between patients. Kidney and central nervous system involvement worsen prognosis. Many patients experience chronic fatigue and joint pain between flares. Cardiovascular disease has become a leading cause of death in lupus patients.",

        # Clinical presentation variant
        "Patients with lupus commonly present with combinations of fatigue, arthralgia or arthritis, and constitutional symptoms like fever and weight loss. Joint pain is often migratory and non-erosive. Skin manifestations include the malar rash, discoid lesions, and photosensitive rashes. Fatigue can be debilitating. Fever may be low-grade and persistent.",
    ]

    for i, content in enumerate(lupus_templates):
        documents.append((f"lupus_doc_{i+1}.txt", content))

    # ----------------------------------------------------
    # CLUSTER 4: Fibromyalgia (5 documents)
    # Widespread pain + fatigue
    # Target intra-cluster similarity: 0.65
    # ----------------------------------------------------
    fibro_templates = [
        # Symptoms document
        "Fibromyalgia causes widespread chronic pain affecting muscles, tendons, and ligaments throughout the body. Patients describe aching, burning, or stabbing sensations that shift location. Profound fatigue is nearly universal. Sleep disturbances include difficulty falling asleep and non-restorative sleep. Cognitive difficulties (fibro fog) affect memory and concentration. Headaches and irritable bowel symptoms are common.",

        # Diagnosis document
        "Fibromyalgia diagnosis is clinical, based on widespread pain for at least 3 months and other characteristic symptoms. Physical examination reveals multiple tender points at specific body locations. Blood tests are normal, serving mainly to exclude other conditions. Widespread pain index and symptom severity scores help quantify disease. No imaging or laboratory test confirms diagnosis.",

        # Risk factors document
        "Fibromyalgia affects women much more than men, typically beginning in middle age. Family history suggests genetic predisposition. Physical or emotional trauma may trigger onset. Certain illnesses like lupus or rheumatoid arthritis increase risk. Stress, anxiety, and depression are associated but causation is unclear.",

        # Treatment document
        "Fibromyalgia treatment is multifaceted, focusing on symptom management. Medications include pregabalin, duloxetine, and milnacipran approved specifically for fibromyalgia. Low-dose tricyclic antidepressants improve sleep and pain. Exercise therapy, despite initial difficulty, provides substantial benefit. Cognitive behavioral therapy helps with coping. Sleep hygiene is important.",

        # Prognosis and complications document
        "Fibromyalgia is chronic but not progressive or life-threatening. Symptoms fluctuate in intensity over time. Many patients experience some improvement with treatment, though complete resolution is rare. Quality of life can be significantly impaired. Work disability may occur in severe cases. Associated mood disorders like depression and anxiety are common and require treatment.",
    ]

    for i, content in enumerate(fibro_templates):
        documents.append((f"fibromyalgia_doc_{i+1}.txt", content))

    # Ground truth metadata
    metadata = {
        'scenario_name': 'symptom_overlap_trap',
        'query': 'Patient presents with chronic fatigue, joint pain, and low-grade fever. What conditions should be considered in differential diagnosis?',
        'clusters': {
            'chronic_fatigue_syndrome': {
                'docs': ['cfs_doc_1.txt', 'cfs_doc_2.txt', 'cfs_doc_3.txt', 'cfs_doc_4.txt',
                        'cfs_doc_5.txt', 'cfs_doc_6.txt', 'cfs_doc_7.txt', 'cfs_doc_8.txt'],
                'target_query_sim': 0.675,
                'primary_symptom': 'fatigue'
            },
            'rheumatoid_arthritis': {
                'docs': ['ra_doc_1.txt', 'ra_doc_2.txt', 'ra_doc_3.txt', 'ra_doc_4.txt',
                        'ra_doc_5.txt', 'ra_doc_6.txt', 'ra_doc_7.txt', 'ra_doc_8.txt'],
                'target_query_sim': 0.70,
                'primary_symptom': 'joint_pain'
            },
            'lupus': {
                'docs': ['lupus_doc_1.txt', 'lupus_doc_2.txt', 'lupus_doc_3.txt', 'lupus_doc_4.txt',
                        'lupus_doc_5.txt', 'lupus_doc_6.txt', 'lupus_doc_7.txt', 'lupus_doc_8.txt'],
                'target_query_sim': 0.735,
                'primary_symptom': 'all_three_balanced',
                'note': 'Highest query similarity - the trap for MMR'
            },
            'fibromyalgia': {
                'docs': ['fibromyalgia_doc_1.txt', 'fibromyalgia_doc_2.txt', 'fibromyalgia_doc_3.txt',
                        'fibromyalgia_doc_4.txt', 'fibromyalgia_doc_5.txt'],
                'target_query_sim': 0.64,
                'primary_symptom': 'pain_fatigue'
            }
        },
        'expected_behavior': {
            'mmr': 'Likely picks 3-4 lupus docs (highest initial score), missing CFS/RA perspectives',
            'qubo': 'Global optimization finds balanced set: 2 lupus + 1-2 RA + 1 CFS + 0-1 fibro'
        }
    }

    return documents, metadata


# ============================================================================
# SCENARIO B: Diagnostic Confusion Trap (Autoimmune Diseases)
# ============================================================================

def generate_diagnostic_confusion_scenario() -> Tuple[List[Tuple[str, str]], Dict[str, any]]:
    """
    Generate hard-case with autoimmune diseases having overlapping presentations.

    Query: "Patient with autoimmune symptoms and multi-organ involvement"

    Cluster Design:
    - Lupus: Kidney, skin, joints
    - Rheumatoid Arthritis: Joints, lungs
    - Scleroderma: Skin, GI, lungs
    - Sjögren's: Glands, joints

    All score similarly on query, creating diverse optimal solution.
    """
    documents = []

    # Lupus - multi-organ autoimmune
    lupus_docs = [
        "Systemic lupus erythematosus is a multi-system autoimmune disease affecting kidneys, skin, joints, and blood. Immune complexes deposit in glomeruli causing lupus nephritis. Malar rash and photosensitivity are characteristic skin findings. Arthritis is common but non-erosive. Hematologic abnormalities include anemia and thrombocytopenia.",

        "Lupus kidney involvement ranges from mild proteinuria to rapidly progressive glomerulonephritis. Skin manifestations include the butterfly rash, discoid lupus, and photosensitive rashes. Joint symptoms mirror rheumatoid arthritis but without erosions. Autoantibodies like anti-dsDNA are disease-specific. Multiple organs are typically affected.",
    ]

    # Rheumatoid Arthritis - joints + extra-articular
    ra_docs = [
        "Rheumatoid arthritis is a systemic autoimmune disease primarily targeting synovial joints but also causing extra-articular manifestations. Interstitial lung disease develops in some patients. Rheumatoid nodules appear subcutaneously. Vasculitis can affect multiple organs. Symmetric polyarthritis is the hallmark presentation.",

        "Joint inflammation in rheumatoid arthritis leads to cartilage and bone erosion. Pulmonary involvement includes interstitial fibrosis and pleural disease. Cardiovascular risk increases significantly. Rheumatoid factor and anti-CCP antibodies indicate autoimmune pathogenesis. Systemic inflammation affects multiple body systems.",
    ]

    # Scleroderma - skin + internal organs
    sclero_docs = [
        "Systemic sclerosis causes skin thickening and internal organ fibrosis. Gastrointestinal involvement leads to dysmotility and malabsorption. Interstitial lung disease is a major cause of mortality. Skin changes progress from edema to sclerosis. Autoantibodies like anti-Scl-70 are associated with diffuse disease.",

        "Scleroderma affects skin, lungs, esophagus, and kidneys through excessive collagen deposition. Raynaud's phenomenon precedes other symptoms. Pulmonary arterial hypertension and interstitial fibrosis cause respiratory compromise. Renal crisis can lead to malignant hypertension. The autoimmune process causes widespread fibrosis.",
    ]

    # Sjögren's - glands + joints
    sjogrens_docs = [
        "Sjögren's syndrome is an autoimmune disease targeting exocrine glands, causing dry eyes and mouth. Arthralgia or arthritis affects many patients. Lymphocytic infiltration destroys lacrimal and salivary glands. Extraglandular manifestations include lung and kidney involvement. Anti-Ro and anti-La antibodies are characteristic.",

        "Autoimmune destruction of glands in Sjögren's syndrome leads to sicca symptoms. Joint pain mimics rheumatoid arthritis. Interstitial lung disease may develop. The immune system attacks glandular epithelium. Multiple organs can be affected beyond glands.",
    ]

    for i, content in enumerate(lupus_docs):
        documents.append((f"lupus_multi_organ_{i+1}.txt", content))
    for i, content in enumerate(ra_docs):
        documents.append((f"ra_extra_articular_{i+1}.txt", content))
    for i, content in enumerate(sclero_docs):
        documents.append((f"scleroderma_{i+1}.txt", content))
    for i, content in enumerate(sjogrens_docs):
        documents.append((f"sjogrens_{i+1}.txt", content))

    metadata = {
        'scenario_name': 'diagnostic_confusion_trap',
        'query': 'Patient presents with autoimmune symptoms and multi-organ involvement. What conditions should be considered?',
        'clusters': {
            'lupus': {'docs': [f"lupus_multi_organ_{i+1}.txt" for i in range(2)], 'organs': ['kidney', 'skin', 'joints']},
            'rheumatoid_arthritis': {'docs': [f"ra_extra_articular_{i+1}.txt" for i in range(2)], 'organs': ['joints', 'lungs']},
            'scleroderma': {'docs': [f"scleroderma_{i+1}.txt" for i in range(2)], 'organs': ['skin', 'GI', 'lungs']},
            'sjogrens': {'docs': [f"sjogrens_{i+1}.txt" for i in range(2)], 'organs': ['glands', 'joints']}
        },
        'expected_behavior': {
            'mmr': 'Picks from first high-scoring cluster, missing other autoimmune perspectives',
            'qubo': 'Selects across diseases for comprehensive differential diagnosis'
        }
    }

    return documents, metadata


# ============================================================================
# Main Generation Functions
# ============================================================================

def save_documents(documents: List[Tuple[str, str]], output_dir: Path) -> None:
    """Save documents to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename, content in documents:
        filepath = output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

    print(f"✓ Saved {len(documents)} documents to {output_dir}")


def generate_all_scenarios(base_dir: str = "data/medical_diagnosis/strategic") -> Dict[str, Dict]:
    """
    Generate all hard-case scenarios.

    Returns:
        Dictionary mapping scenario names to metadata
    """
    base_path = Path(base_dir)
    all_metadata = {}

    # Scenario A: Symptom Overlap Trap
    print("\n" + "="*60)
    print("Generating Scenario A: Symptom Overlap Trap")
    print("="*60)
    docs_a, meta_a = generate_symptom_overlap_scenario()
    save_documents(docs_a, base_path / "scenario_a_symptom_overlap")
    all_metadata['scenario_a'] = meta_a
    print(f"  - Created {len(docs_a)} documents across 4 disease clusters")
    print(f"  - Query: {meta_a['query']}")

    # Scenario B: Diagnostic Confusion
    print("\n" + "="*60)
    print("Generating Scenario B: Diagnostic Confusion Trap")
    print("="*60)
    docs_b, meta_b = generate_diagnostic_confusion_scenario()
    save_documents(docs_b, base_path / "scenario_b_diagnostic_confusion")
    all_metadata['scenario_b'] = meta_b
    print(f"  - Created {len(docs_b)} documents across 4 autoimmune diseases")
    print(f"  - Query: {meta_b['query']}")

    # Save metadata
    import json
    metadata_file = base_path / "scenarios_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(all_metadata, f, indent=2)
    print(f"\n✓ Saved metadata to {metadata_file}")

    return all_metadata


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Strategic Synthetic Data Generator for QUBO-RAG")
    print("="*60)
    print("\nThis generator creates 'similarity cluster traps' to showcase")
    print("QUBO-RAG's competitive advantage over MMR.")
    print("\nKey Design:")
    print("  - Moderate intra-cluster similarity (0.65-0.80)")
    print("  - Multiple high-relevance clusters")
    print("  - MMR gets trapped, QUBO escapes via global optimization")

    metadata = generate_all_scenarios()

    print("\n" + "="*60)
    print("Generation Complete!")
    print("="*60)
    print("\nNext Steps:")
    print("  1. Run validate_similarity_structure.py to verify targets")
    print("  2. Index documents in vector database")
    print("  3. Run comparison: Naive vs MMR vs QUBO")
    print("  4. Observe QUBO's superior cluster coverage")
