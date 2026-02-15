/**
 * Mock data for demo when backend is unavailable
 */

import type { CompareResponse, MethodResult, UMAPPoint } from './api';

// Dataset-specific cluster configurations
const DATASET_CLUSTERS: Record<string, { name: string; cx: number; cy: number }[]> = {
  medical: [
    { name: 'mononucleosis', cx: -2, cy: 1 },
    { name: 'lupus', cx: 2, cy: 2 },
    { name: 'lyme', cx: 1, cy: -2 },
    { name: 'fibromyalgia', cx: -1, cy: -1 },
    { name: 'chronic_fatigue', cx: 0, cy: 3 },
  ],
  legal: [
    { name: 'contract_breach', cx: -2, cy: 1 },
    { name: 'employment_discrimination', cx: 2, cy: 2 },
    { name: 'ip_patent', cx: 1, cy: -2 },
    { name: 'personal_injury', cx: -1, cy: -1 },
    { name: 'criminal_procedure', cx: 0, cy: 3 },
  ],
  greedy_trap: [
    { name: 'cluster_A', cx: -1.5, cy: 0 },
    { name: 'cluster_B', cx: 1.5, cy: 0 },
    { name: 'cluster_C', cx: 0, cy: 1.5 },
    { name: 'cluster_D', cx: 0, cy: -1.5 },
    { name: 'cluster_E', cx: 0, cy: 0 },
  ],
};

// Mock UMAP points for visualization
const generateMockUMAPPoints = (dataset: string): UMAPPoint[] => {
  const clusters = DATASET_CLUSTERS[dataset] || DATASET_CLUSTERS.medical;

  const points: UMAPPoint[] = [];
  let id = 0;

  clusters.forEach(cluster => {
    const count = 5 + Math.floor(Math.random() * 4);
    for (let i = 0; i < count; i++) {
      points.push({
        x: cluster.cx + (Math.random() - 0.5) * 1.5,
        y: cluster.cy + (Math.random() - 0.5) * 1.5,
        chunk_id: `chunk_${id++}`,
        source: `${cluster.name}_doc_${i + 1}.txt`,
        cluster: cluster.name,
        is_selected: false,
        selected_by: [],
      });
    }
  });

  return points;
};

// Dataset-specific mock results
const MOCK_DATA: Record<string, { topk: MethodResult; mmr: MethodResult; qubo: MethodResult }> = {
  medical: {
    topk: {
      method: 'topk',
      results: [
        { rank: 1, score: 0.92, text: 'Infectious mononucleosis typically presents with fatigue, sore throat, and swollen lymph nodes. Patients often experience extreme tiredness lasting several weeks.', source: 'mononucleosis_doc_1.txt', chunk_id: 'chunk_0' },
        { rank: 2, score: 0.91, text: 'Mono symptoms include persistent fatigue, fever, and joint pain. The Epstein-Barr virus is the most common cause of infectious mononucleosis.', source: 'mononucleosis_doc_2.txt', chunk_id: 'chunk_1' },
        { rank: 3, score: 0.89, text: 'Mononucleosis causes significant fatigue and malaise. Patients may experience muscle aches and low-grade fever for extended periods.', source: 'mononucleosis_doc_3.txt', chunk_id: 'chunk_2' },
        { rank: 4, score: 0.88, text: 'The glandular fever virus leads to chronic tiredness and joint discomfort. Recovery from mononucleosis can take several months.', source: 'mononucleosis_doc_4.txt', chunk_id: 'chunk_3' },
        { rank: 5, score: 0.87, text: 'EBV infection manifests as extreme exhaustion with occasional febrile episodes. Splenic enlargement may occur in severe cases.', source: 'mononucleosis_doc_5.txt', chunk_id: 'chunk_4' },
      ],
      metrics: { latency_ms: 23, intra_list_similarity: 0.72, cluster_coverage: 1, total_clusters: 5, avg_relevance: 0.894 },
      llm_response: 'Based on the retrieved documents, the symptoms suggest infectious mononucleosis (mono). The patient should be tested for Epstein-Barr virus. Note: This response is limited because all retrieved documents discuss only mononucleosis.',
    },
    mmr: {
      method: 'mmr',
      results: [
        { rank: 1, score: 0.92, text: 'Infectious mononucleosis typically presents with fatigue, sore throat, and swollen lymph nodes. Patients often experience extreme tiredness lasting several weeks.', source: 'mononucleosis_doc_1.txt', chunk_id: 'chunk_0' },
        { rank: 2, score: 0.78, text: 'Systemic lupus erythematosus (SLE) causes fatigue, joint pain, and low-grade fever. Autoimmune inflammation affects multiple organ systems.', source: 'lupus_doc_1.txt', chunk_id: 'chunk_10' },
        { rank: 3, score: 0.75, text: 'Lyme disease from tick bites causes fatigue, joint pain, and fever. Early treatment with antibiotics is essential.', source: 'lyme_doc_1.txt', chunk_id: 'chunk_15' },
        { rank: 4, score: 0.88, text: 'Mono symptoms include persistent fatigue, fever, and joint pain. The Epstein-Barr virus is the most common cause.', source: 'mononucleosis_doc_2.txt', chunk_id: 'chunk_1' },
        { rank: 5, score: 0.71, text: 'Chronic fatigue syndrome presents with debilitating tiredness not improved by rest. Cognitive difficulties are common.', source: 'chronic_fatigue_doc_1.txt', chunk_id: 'chunk_25' },
      ],
      metrics: { latency_ms: 45, intra_list_similarity: 0.48, cluster_coverage: 4, total_clusters: 5, avg_relevance: 0.808 },
      llm_response: 'The symptoms could indicate several conditions: mononucleosis, lupus, Lyme disease, or chronic fatigue syndrome. Further testing is recommended to differentiate between these diagnoses.',
    },
    qubo: {
      method: 'qubo',
      results: [
        { rank: 1, score: 0.92, text: 'Infectious mononucleosis typically presents with fatigue, sore throat, and swollen lymph nodes. Patients often experience extreme tiredness lasting several weeks.', source: 'mononucleosis_doc_1.txt', chunk_id: 'chunk_0' },
        { rank: 2, score: 0.78, text: 'Systemic lupus erythematosus (SLE) causes fatigue, joint pain, and low-grade fever. Characteristic butterfly rash and photosensitivity may be present.', source: 'lupus_doc_1.txt', chunk_id: 'chunk_10' },
        { rank: 3, score: 0.75, text: 'Lyme disease from tick bites causes fatigue, joint pain, and fever. Bulls-eye rash (erythema migrans) is a distinctive early sign.', source: 'lyme_doc_1.txt', chunk_id: 'chunk_15' },
        { rank: 4, score: 0.73, text: 'Fibromyalgia causes widespread musculoskeletal pain with fatigue, sleep problems, and cognitive difficulties. Tender points are characteristic.', source: 'fibromyalgia_doc_1.txt', chunk_id: 'chunk_20' },
        { rank: 5, score: 0.71, text: 'Chronic fatigue syndrome presents with debilitating tiredness not improved by rest. Post-exertional malaise is a key diagnostic criterion.', source: 'chronic_fatigue_doc_1.txt', chunk_id: 'chunk_25' },
      ],
      metrics: { latency_ms: 87, intra_list_similarity: 0.31, cluster_coverage: 5, total_clusters: 5, avg_relevance: 0.778 },
      llm_response: 'The symptoms warrant evaluation for multiple conditions:\n\n1. **Mononucleosis** - Check for EBV antibodies\n2. **Lupus** - ANA test, look for butterfly rash\n3. **Lyme disease** - Check for tick exposure, order Lyme titers\n4. **Fibromyalgia** - Assess tender points\n5. **Chronic fatigue syndrome** - Rule out other causes first\n\nComprehensive differential diagnosis enables targeted testing.',
    },
  },
  legal: {
    topk: {
      method: 'topk',
      results: [
        { rank: 1, score: 0.91, text: 'In Smith v. TechCorp (2021), the court ruled that unpaid bonuses constituted breach of employment contract when bonus criteria were clearly met by the employee.', source: 'contract_breach_doc_1.txt', chunk_id: 'chunk_0' },
        { rank: 2, score: 0.89, text: 'Johnson v. Acme Industries established that verbal promises of compensation can form binding contracts when supported by documented performance metrics.', source: 'contract_breach_doc_2.txt', chunk_id: 'chunk_1' },
        { rank: 3, score: 0.88, text: 'The precedent in Williams v. MegaCorp requires employers to honor bonus structures outlined in offer letters, even without formal contract language.', source: 'contract_breach_doc_3.txt', chunk_id: 'chunk_2' },
        { rank: 4, score: 0.87, text: 'Martinez v. StartupXYZ found that discretionary bonus clauses do not exempt employers from paying when performance targets are demonstrably achieved.', source: 'contract_breach_doc_4.txt', chunk_id: 'chunk_3' },
        { rank: 5, score: 0.85, text: 'In Davis v. Financial Services Ltd, failure to pay promised retention bonuses was deemed material breach warranting compensatory damages.', source: 'contract_breach_doc_5.txt', chunk_id: 'chunk_4' },
      ],
      metrics: { latency_ms: 21, intra_list_similarity: 0.74, cluster_coverage: 1, total_clusters: 5, avg_relevance: 0.88 },
      llm_response: 'Based on the retrieved cases, all precedents relate to contract breach for unpaid bonuses. The cases consistently support employee claims when bonus criteria are documented. However, this analysis only covers breach of contract precedents and misses other relevant legal theories.',
    },
    mmr: {
      method: 'mmr',
      results: [
        { rank: 1, score: 0.91, text: 'In Smith v. TechCorp (2021), the court ruled that unpaid bonuses constituted breach of employment contract when bonus criteria were clearly met by the employee.', source: 'contract_breach_doc_1.txt', chunk_id: 'chunk_0' },
        { rank: 2, score: 0.76, text: 'Thompson v. GlobalBank established that bonus denials based on protected characteristics constitute employment discrimination under Title VII.', source: 'employment_discrimination_doc_1.txt', chunk_id: 'chunk_10' },
        { rank: 3, score: 0.73, text: 'The EEOC guidelines in Rodriguez v. Manufacturing Inc clarify that disparate impact in bonus distribution creates actionable discrimination claims.', source: 'employment_discrimination_doc_2.txt', chunk_id: 'chunk_11' },
        { rank: 4, score: 0.87, text: 'Martinez v. StartupXYZ found that discretionary bonus clauses do not exempt employers from paying when performance targets are demonstrably achieved.', source: 'contract_breach_doc_4.txt', chunk_id: 'chunk_3' },
        { rank: 5, score: 0.69, text: 'In Parker v. Insurance Corp, the court awarded punitive damages when bonus withholding was found to be retaliatory for whistleblower complaints.', source: 'employment_discrimination_doc_3.txt', chunk_id: 'chunk_12' },
      ],
      metrics: { latency_ms: 43, intra_list_similarity: 0.51, cluster_coverage: 2, total_clusters: 5, avg_relevance: 0.792 },
      llm_response: 'The cases span contract breach and employment discrimination. Consider both breach of contract claims under Smith v. TechCorp and potential discrimination claims if bonus denial affected protected classes. MMR found some diversity but missed other relevant areas.',
    },
    qubo: {
      method: 'qubo',
      results: [
        { rank: 1, score: 0.91, text: 'In Smith v. TechCorp (2021), the court ruled that unpaid bonuses constituted breach of employment contract when bonus criteria were clearly met by the employee.', source: 'contract_breach_doc_1.txt', chunk_id: 'chunk_0' },
        { rank: 2, score: 0.76, text: 'Thompson v. GlobalBank established that bonus denials based on protected characteristics constitute employment discrimination under Title VII.', source: 'employment_discrimination_doc_1.txt', chunk_id: 'chunk_10' },
        { rank: 3, score: 0.72, text: 'Wilson v. Tech Innovations found that bonus structures tied to IP development create implied licensing rights when compensation is withheld.', source: 'ip_patent_doc_1.txt', chunk_id: 'chunk_15' },
        { rank: 4, score: 0.68, text: 'In Chen v. Consulting LLC, emotional distress damages were awarded alongside contract damages when bonus withholding caused documented psychological harm.', source: 'personal_injury_doc_1.txt', chunk_id: 'chunk_20' },
        { rank: 5, score: 0.65, text: 'Federal sentencing guidelines in US v. Morrison address criminal fraud charges when systematic bonus withholding constitutes wage theft schemes.', source: 'criminal_procedure_doc_1.txt', chunk_id: 'chunk_25' },
      ],
      metrics: { latency_ms: 82, intra_list_similarity: 0.29, cluster_coverage: 5, total_clusters: 5, avg_relevance: 0.744 },
      llm_response: 'Comprehensive legal analysis reveals multiple avenues:\n\n1. **Contract Breach** - Smith v. TechCorp precedent for documented bonus criteria\n2. **Employment Discrimination** - Thompson v. GlobalBank if protected class affected\n3. **IP Rights** - Wilson v. Tech Innovations if bonus tied to IP work\n4. **Emotional Distress** - Chen v. Consulting LLC for psychological harm damages\n5. **Criminal Fraud** - US v. Morrison if systematic wage theft pattern exists\n\nQUBO retrieval enables comprehensive multi-theory legal strategy.',
    },
  },
  greedy_trap: {
    topk: {
      method: 'topk',
      results: [
        { rank: 1, score: 0.95, text: 'Cluster A Document 1: High relevance decoy content designed to attract greedy selection algorithms. Contains keywords matching common queries.', source: 'cluster_A_doc_1.txt', chunk_id: 'chunk_0' },
        { rank: 2, score: 0.94, text: 'Cluster A Document 2: Nearly identical content to maximize similarity scores. Greedy algorithms will select this due to high relevance.', source: 'cluster_A_doc_2.txt', chunk_id: 'chunk_1' },
        { rank: 3, score: 0.93, text: 'Cluster A Document 3: Third highly similar document. Demonstrates how Top-K fills results with redundant information.', source: 'cluster_A_doc_3.txt', chunk_id: 'chunk_2' },
        { rank: 4, score: 0.92, text: 'Cluster A Document 4: Continues the pattern of high-scoring but redundant content. Information gain is minimal.', source: 'cluster_A_doc_4.txt', chunk_id: 'chunk_3' },
        { rank: 5, score: 0.91, text: 'Cluster A Document 5: Final redundant selection. All five results from same cluster despite diverse alternatives existing.', source: 'cluster_A_doc_5.txt', chunk_id: 'chunk_4' },
      ],
      metrics: { latency_ms: 18, intra_list_similarity: 0.89, cluster_coverage: 1, total_clusters: 5, avg_relevance: 0.93 },
      llm_response: 'All retrieved documents contain essentially the same information from Cluster A. Despite high relevance scores, the response can only address one perspective. This demonstrates the greedy trap where Top-K selection fails.',
    },
    mmr: {
      method: 'mmr',
      results: [
        { rank: 1, score: 0.95, text: 'Cluster A Document 1: High relevance decoy content designed to attract greedy selection algorithms. Contains keywords matching common queries.', source: 'cluster_A_doc_1.txt', chunk_id: 'chunk_0' },
        { rank: 2, score: 0.72, text: 'Cluster B Document 1: Alternative perspective with moderate relevance. MMR may select this to increase diversity.', source: 'cluster_B_doc_1.txt', chunk_id: 'chunk_7' },
        { rank: 3, score: 0.93, text: 'Cluster A Document 3: MMR falls back to high-relevance cluster when diversity penalty is insufficient.', source: 'cluster_A_doc_3.txt', chunk_id: 'chunk_2' },
        { rank: 4, score: 0.68, text: 'Cluster C Document 1: Third cluster represented. MMR improves over Top-K but misses optimal global solution.', source: 'cluster_C_doc_1.txt', chunk_id: 'chunk_14' },
        { rank: 5, score: 0.91, text: 'Cluster A Document 5: MMR returns to dominant cluster. Greedy sequential selection limits diversity.', source: 'cluster_A_doc_5.txt', chunk_id: 'chunk_4' },
      ],
      metrics: { latency_ms: 41, intra_list_similarity: 0.58, cluster_coverage: 3, total_clusters: 5, avg_relevance: 0.838 },
      llm_response: 'MMR improved diversity by including Clusters A, B, and C. However, greedy sequential selection still returns to the dominant cluster, missing Clusters D and E. The local optimization trap prevents optimal coverage.',
    },
    qubo: {
      method: 'qubo',
      results: [
        { rank: 1, score: 0.95, text: 'Cluster A Document 1: High relevance decoy content designed to attract greedy selection algorithms. Contains keywords matching common queries.', source: 'cluster_A_doc_1.txt', chunk_id: 'chunk_0' },
        { rank: 2, score: 0.72, text: 'Cluster B Document 1: Alternative perspective with moderate relevance. Provides unique information not in Cluster A.', source: 'cluster_B_doc_1.txt', chunk_id: 'chunk_7' },
        { rank: 3, score: 0.68, text: 'Cluster C Document 1: Third distinct perspective. QUBO global optimization selects for maximum information coverage.', source: 'cluster_C_doc_1.txt', chunk_id: 'chunk_14' },
        { rank: 4, score: 0.65, text: 'Cluster D Document 1: Fourth unique cluster. QUBO considers all combinations simultaneously to find optimal set.', source: 'cluster_D_doc_1.txt', chunk_id: 'chunk_21' },
        { rank: 5, score: 0.62, text: 'Cluster E Document 1: Complete coverage achieved. QUBO sacrifices some relevance for maximum diversity and information gain.', source: 'cluster_E_doc_1.txt', chunk_id: 'chunk_28' },
      ],
      metrics: { latency_ms: 94, intra_list_similarity: 0.18, cluster_coverage: 5, total_clusters: 5, avg_relevance: 0.724 },
      llm_response: 'QUBO achieves complete cluster coverage by optimizing globally:\n\n1. **Cluster A** - Primary high-relevance content\n2. **Cluster B** - Alternative perspective\n3. **Cluster C** - Third distinct viewpoint\n4. **Cluster D** - Fourth unique angle\n5. **Cluster E** - Complete coverage\n\nGlobal optimization escapes the greedy trap, providing comprehensive information despite adversarial data distribution.',
    },
  },
};

export function getMockCompareResponse(query: string, dataset: string): CompareResponse {
  const datasetKey = dataset in MOCK_DATA ? dataset : 'medical';
  const mockData = MOCK_DATA[datasetKey];
  const umapPoints = generateMockUMAPPoints(datasetKey);

  // Mark selected points based on mock results
  const topkIds = mockData.topk.results.map(r => r.chunk_id);
  const mmrIds = mockData.mmr.results.map(r => r.chunk_id);
  const quboIds = mockData.qubo.results.map(r => r.chunk_id);

  umapPoints.forEach(point => {
    const selectedBy: string[] = [];
    if (topkIds.includes(point.chunk_id)) selectedBy.push('topk');
    if (mmrIds.includes(point.chunk_id)) selectedBy.push('mmr');
    if (quboIds.includes(point.chunk_id)) selectedBy.push('qubo');

    if (selectedBy.length > 0) {
      point.is_selected = true;
      point.selected_by = selectedBy;
    }
  });

  return {
    query,
    dataset,
    topk: mockData.topk,
    mmr: mockData.mmr,
    qubo: mockData.qubo,
    umap_points: umapPoints,
    query_point: { x: 0.5, y: 0.5 },
  };
}

export const DEMO_QUERIES = {
  medical: [
    'Patient with chronic fatigue, joint pain, and occasional low-grade fever',
    'Extreme tiredness, difficulty concentrating, and muscle weakness',
    'Persistent exhaustion with cognitive difficulties and widespread pain',
    'Fatigue with joint stiffness, rash, and sensitivity to sunlight',
    'Bilateral joint swelling with morning stiffness and systemic inflammation',
    'Sudden onset tremor with cognitive decline and muscle rigidity',
  ],
  greedy_trap: [
    'Patient experiencing persistent fatigue with multiple symptoms',
    'General tiredness affecting multiple body systems',
    'Diagnostic challenge with overlapping symptom clusters',
  ],
  legal: [
    'Employment contract breach involving unpaid bonuses',
    'Discrimination claim in workplace termination',
    'Intellectual property dispute in software development',
    'Corporate liability issues in employee compensation disputes',
    'Unauthorized use of proprietary information by former employees',
  ],
  wikipedia: [
    'How do greenhouse gases and carbon emissions contribute to climate change, and what renewable energy solutions can mitigate environmental damage?',
    'Explain quantum mechanical principles and their applications in modern quantum computing and field theory',
    'What are the foundational technologies behind artificial intelligence, machine learning, and neural networks, and how do they enable modern robotics?',
    'How does the immune system respond to diseases like COVID-19 and what role do vaccines play in public health?',
    'Trace the evolution of democracy, human rights, and major civil rights movements including women\'s suffrage and feminism',
    'Evaluate modern transportation systems including rail transit, electric vehicles, and high-speed infrastructure',
  ],
};
