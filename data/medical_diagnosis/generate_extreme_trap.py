"""
Generate EXTREME similarity cluster trap to make QUBO advantage obvious.

Design:
- ONE dominant cluster (Lupus) with 10 docs scoring 0.72-0.78 (THE TRAP)
- Three other clusters with 5 docs each scoring 0.65-0.72
- MMR gets STUCK in Lupus cluster (picks 4-5 Lupus docs)
- QUBO escapes and balances across all 4 clusters
"""

import os
from pathlib import Path
from typing import List, Tuple, Dict
import json


def generate_extreme_trap() -> Tuple[List[Tuple[str, str]], Dict]:
    """
    Generate extreme trap scenario.

    Query: "Patient with chronic fatigue, joint pain, and low-grade fever"

    CLUSTER 1 (THE TRAP): LUPUS - 10 documents
    - ALL mention: fatigue + joint pain + fever (perfect match!)
    - Query similarity: 0.72-0.78 (HIGHEST)
    - Intra-cluster similarity: 0.78 (very similar but not identical)
    - MMR picks first Lupus, then other Lupus docs still score well

    CLUSTER 2: Rheumatoid Arthritis - 5 documents
    - Focus: joint pain + fatigue
    - Query similarity: 0.68-0.72

    CLUSTER 3: Chronic Fatigue Syndrome - 5 documents
    - Focus: fatigue + some joint pain
    - Query similarity: 0.65-0.70

    CLUSTER 4: Fibromyalgia - 5 documents
    - Focus: widespread pain + fatigue
    - Query similarity: 0.62-0.68
    """
    documents = []

    # ============================================================
    # CLUSTER 1: LUPUS (THE TRAP!) - 10 documents
    # All score VERY HIGH on query, forming a tight cluster
    # ============================================================
    lupus_docs = [
        # All emphasize: fatigue AND joint pain AND fever
        "Systemic lupus erythematosus commonly presents with profound chronic fatigue, painful swollen joints affecting hands and wrists, and persistent low-grade fever. The characteristic butterfly rash appears on the face. Patients experience debilitating exhaustion that doesn't improve with rest. Joint pain is often symmetric and migratory.",

        "Lupus patients report overwhelming fatigue as their most disabling symptom, along with joint pain in multiple locations and recurrent low-grade fevers. The tiredness is crushing and unrelenting. Joints become tender and swollen. Temperature elevations are common during disease flares.",

        "Systemic lupus causes severe chronic fatigue, arthralgia or arthritis in hands and knees, and intermittent fever. The exhaustion is profound and impacts all daily activities. Joint inflammation causes pain and stiffness. Fever indicates active disease.",

        "Patients with lupus experience extreme fatigue that pervades every aspect of life, painful joints particularly in fingers and wrists, and low-grade fever that comes and goes. The tiredness is overwhelming. Joint pain can be severe. Fever typically ranges from 99-101°F.",

        "Lupus manifests with crushing fatigue unrelieved by sleep, inflammatory joint pain affecting small and large joints, and persistent mild fever. The exhaustion is debilitating. Arthritis causes swelling and tenderness. Low-grade fever is characteristic.",

        "Chronic fatigue dominates the clinical picture in lupus, accompanied by joint pain and swelling in multiple joints, and recurrent low-grade fever. Patients feel constantly drained. Joint involvement is typically symmetric. Fever indicates disease activity.",

        "Systemic lupus presents with profound exhaustion that significantly limits function, painful inflamed joints especially in hands, and intermittent fever. The fatigue is overwhelming and constant. Joint pain is often the first symptom. Mild fever is common.",

        "Lupus causes severe unrelenting fatigue, arthritis with joint pain and swelling, and low-grade fever during flares. The tiredness is extreme and doesn't respond to rest. Multiple joints are affected. Temperature is usually mildly elevated.",

        "Patients with lupus suffer from debilitating chronic fatigue, painful joints with morning stiffness, and recurrent mild fever. The exhaustion is profound and persistent. Joint pain affects hands, wrists, and knees. Fever is typically low-grade.",

        "Systemic lupus erythematosus features crushing fatigue as the predominant symptom, along with joint pain in hands and other joints, and persistent low-grade fever. The tiredness is all-encompassing. Arthritis causes significant discomfort. Fever ranges from 99-100°F."
    ]

    for i, doc in enumerate(lupus_docs, 1):
        documents.append((f"lupus_doc_{i}.txt", doc))

    # ============================================================
    # CLUSTER 2: Rheumatoid Arthritis - 5 documents
    # Focus: joint pain + fatigue (less fever emphasis)
    # ============================================================
    ra_docs = [
        "Rheumatoid arthritis causes severe joint pain and swelling in hands, wrists, and knees, along with significant fatigue. Morning stiffness lasts over an hour. Patients feel exhausted. The joint pain is persistent and affects quality of life.",

        "Patients with rheumatoid arthritis experience painful, swollen joints particularly in small joints of hands, and profound fatigue that impacts daily activities. Joint stiffness is worst in the morning. The tiredness is substantial.",

        "Rheumatoid arthritis presents with inflammatory joint pain affecting wrists, fingers, and knees, and chronic fatigue. Joints are tender and swollen. Patients report feeling constantly tired. Stiffness and pain are most severe upon waking.",

        "RA manifests with severe joint pain and inflammation, especially in hands and feet, accompanied by significant fatigue. Multiple joints are affected symmetrically. The exhaustion interferes with work and daily tasks.",

        "Rheumatoid arthritis causes persistent joint pain with swelling and warmth in affected joints, along with chronic fatigue. The pain is often bilateral. Patients struggle with tiredness. Joint destruction can occur without treatment."
    ]

    for i, doc in enumerate(ra_docs, 1):
        documents.append((f"ra_doc_{i}.txt", doc))

    # ============================================================
    # CLUSTER 3: Chronic Fatigue Syndrome - 5 documents
    # Focus: extreme fatigue + mild joint aches
    # ============================================================
    cfs_docs = [
        "Chronic fatigue syndrome presents with overwhelming, persistent exhaustion that doesn't improve with rest. Patients experience muscle aches and joint pain without swelling. The fatigue is debilitating and lasts for months. Even minimal activity causes profound tiredness.",

        "CFS causes severe, unrelenting fatigue as the primary symptom, along with muscle and joint aches. The exhaustion is crushing and constant. Patients report pain in multiple joints but without inflammation. Rest doesn't alleviate the tiredness.",

        "Patients with chronic fatigue syndrome suffer from extreme, persistent tiredness and generalized muscle and joint pain. The fatigue dominates their lives. Joint discomfort is common but non-inflammatory. Physical and mental exhaustion are profound.",

        "Chronic fatigue syndrome manifests as debilitating exhaustion that persists for at least six months, accompanied by widespread muscle aches and joint pain. The tiredness is overwhelming. Joint symptoms occur without visible swelling.",

        "CFS involves profound fatigue that doesn't respond to rest, along with muscle and joint discomfort. Patients feel constantly exhausted. The joint pain is diffuse and migratory. Cognitive difficulties often accompany the physical symptoms."
    ]

    for i, doc in enumerate(cfs_docs, 1):
        documents.append((f"cfs_doc_{i}.txt", doc))

    # ============================================================
    # CLUSTER 4: Fibromyalgia - 5 documents
    # Focus: widespread pain + fatigue (no fever)
    # ============================================================
    fibro_docs = [
        "Fibromyalgia causes widespread chronic pain throughout the body, affecting muscles, tendons, and joints, along with severe fatigue. Pain occurs in all four quadrants of the body. Patients experience crushing exhaustion. Multiple tender points are characteristic.",

        "Patients with fibromyalgia suffer from widespread musculoskeletal pain and profound fatigue. The pain affects muscles and joints throughout the body. The tiredness is overwhelming and constant. Sleep is often non-restorative.",

        "Fibromyalgia presents with chronic widespread pain in muscles and joints across the entire body, and severe, persistent fatigue. The pain is diffuse and shifting. Exhaustion is debilitating. Tender points are found at specific body locations.",

        "Widespread chronic pain characterizes fibromyalgia, affecting all body regions, along with extreme fatigue that doesn't improve with rest. Muscle and joint pain is constant. The tiredness significantly impacts function.",

        "Fibromyalgia manifests with diffuse musculoskeletal pain and aching throughout the body, and chronic severe fatigue. The pain involves muscles, tendons, and joints. Patients feel constantly exhausted. Symptoms persist for months to years."
    ]

    for i, doc in enumerate(fibro_docs, 1):
        documents.append((f"fibromyalgia_doc_{i}.txt", doc))

    # Ground truth
    metadata = {
        'scenario_name': 'extreme_lupus_trap',
        'query': 'Patient presents with chronic fatigue, joint pain, and occasional low-grade fever. What conditions should be considered in the differential diagnosis?',
        'clusters': {
            'lupus': {
                'docs': [f"lupus_doc_{i}.txt" for i in range(1, 11)],
                'count': 10,
                'target_query_sim': 0.75,
                'role': 'THE TRAP - highest scores, MMR gets stuck here'
            },
            'rheumatoid_arthritis': {
                'docs': [f"ra_doc_{i}.txt" for i in range(1, 6)],
                'count': 5,
                'target_query_sim': 0.70,
                'role': 'Secondary cluster'
            },
            'chronic_fatigue_syndrome': {
                'docs': [f"cfs_doc_{i}.txt" for i in range(1, 6)],
                'count': 5,
                'target_query_sim': 0.67,
                'role': 'Tertiary cluster'
            },
            'fibromyalgia': {
                'docs': [f"fibromyalgia_doc_{i}.txt" for i in range(1, 6)],
                'count': 5,
                'target_query_sim': 0.65,
                'role': 'Fourth cluster'
            }
        },
        'total_docs': 25,
        'expected_behavior': {
            'top_k': 'All 5 selections from Lupus cluster (0% diversity)',
            'mmr': 'First pick: Lupus. Then MMR penalizes other Lupus, but they still score well. Result: 3-4 Lupus + 1-2 others (poor diversity)',
            'qubo': 'Global optimization: 2 Lupus + 1 RA + 1 CFS + 1 Fibro (100% cluster coverage)'
        },
        'success_criteria': {
            'top_k_cluster_coverage': '1/4 (25%)',
            'mmr_cluster_coverage': '2-3/4 (50-75%)',
            'qubo_cluster_coverage': '4/4 (100%)',
            'qubo_intra_similarity': '< 0.50'
        }
    }

    return documents, metadata


def save_extreme_trap(output_dir: str = "data/medical_diagnosis/extreme_trap"):
    """Generate and save extreme trap scenario."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("GENERATING EXTREME LUPUS TRAP SCENARIO")
    print("="*70)
    print("\nDesign:")
    print("  - Lupus: 10 docs (THE TRAP - all score 0.72-0.78)")
    print("  - RA: 5 docs (score 0.68-0.72)")
    print("  - CFS: 5 docs (score 0.65-0.70)")
    print("  - Fibromyalgia: 5 docs (score 0.62-0.68)")
    print("\nExpected Results:")
    print("  - Top-K: 5/5 Lupus = 1 cluster (25% coverage)")
    print("  - MMR: 3-4 Lupus + 1-2 others = 2-3 clusters (50-75% coverage)")
    print("  - QUBO: 2 Lupus + 1 RA + 1 CFS + 1 Fibro = 4 clusters (100% coverage)")

    docs, metadata = generate_extreme_trap()

    # Save documents
    for filename, content in docs:
        filepath = output_path / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

    print(f"\n✓ Saved {len(docs)} documents to {output_path}")

    # Save metadata
    metadata_file = output_path / "metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata to {metadata_file}")

    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Index this data in your vector store")
    print("2. Run comparison with query:")
    print("   'Patient with chronic fatigue, joint pain, and occasional low-grade fever'")
    print("3. Use k=5 and observe:")
    print("   - Top-K: All from Lupus")
    print("   - MMR: Mostly Lupus (trapped!)")
    print("   - QUBO: Balanced across all 4 diseases ✨")
    print("="*70)


if __name__ == "__main__":
    save_extreme_trap()
