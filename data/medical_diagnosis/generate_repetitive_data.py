"""
Generate LARGE REPETITIVE medical diagnosis dataset to showcase QUBO-RAG diversity advantage.

Target: 200+ documents with strategic repetition
Design: Multiple redundant clusters force Naive to grab duplicates while QUBO finds diversity
"""

import os
from pathlib import Path
from typing import List, Tuple


def generate_chronic_fatigue_variants(n=30) -> List[Tuple[str, str]]:
    """Generate 30 highly similar chronic fatigue syndrome documents."""
    templates = [
        "Chronic fatigue syndrome manifests as persistent, overwhelming tiredness that doesn't improve with rest. Patients report severe exhaustion lasting months, difficulty concentrating, and unrefreshing sleep. This debilitating condition significantly impacts daily activities and quality of life.",
        "Persistent fatigue characterizes chronic fatigue syndrome, with patients experiencing profound exhaustion that rest doesn't alleviate. The tiredness is accompanied by cognitive difficulties and non-restorative sleep patterns. Daily functioning becomes severely compromised.",
        "Chronic fatigue syndrome presents with relentless tiredness and exhaustion that persists despite adequate rest. Affected individuals suffer from mental fog, poor sleep quality, and dramatic reduction in normal activities.",
        "Overwhelming exhaustion defines chronic fatigue syndrome, where rest provides no relief. Patients describe crushing tiredness, impaired concentration, and sleep that doesn't refresh. The condition severely limits work and social activities.",
        "Chronic fatigue syndrome involves extreme, unrelenting tiredness that doesn't respond to rest or sleep. Mental fatigue accompanies physical exhaustion, with patients unable to perform routine tasks.",
        "Unremitting exhaustion characterizes chronic fatigue syndrome, with tiredness that pervades every aspect of life. Rest and sleep fail to restore energy. Cognitive function suffers alongside physical stamina.",
        "Chronic fatigue syndrome manifests as severe, persistent exhaustion unrelieved by rest. Patients experience debilitating tiredness, mental cloudiness, and poor sleep. Daily activities become increasingly difficult.",
        "Extreme tiredness dominates chronic fatigue syndrome, continuing despite rest and sleep. The exhaustion is overwhelming, accompanied by cognitive impairment and unrefreshing rest.",
        "Chronic fatigue syndrome presents with crushing, persistent tiredness that rest cannot improve. Mental and physical exhaustion combine to severely limit functioning.",
        "Profound exhaustion characterizes chronic fatigue syndrome, with fatigue persisting regardless of rest. Patients report all-encompassing tiredness, difficulty thinking clearly, and sleep that doesn't help.",
        "Chronic fatigue syndrome involves overwhelming, constant tiredness unresponsive to rest. The exhaustion is profound and affects both body and mind. Normal activities become extremely challenging.",
        "Persistent, severe exhaustion defines chronic fatigue syndrome, with no improvement from rest or sleep. Patients struggle with crushing fatigue, poor concentration, and non-restorative rest.",
        "Chronic fatigue syndrome manifests through relentless, extreme tiredness that rest doesn't resolve. Mental and physical fatigue combine to create profound disability.",
        "Overwhelming, persistent exhaustion characterizes chronic fatigue syndrome. Despite rest and sleep, tiredness remains severe and debilitating. Cognitive function and physical capacity both decline.",
        "Chronic fatigue syndrome presents with severe, unrelenting tiredness that continues despite adequate rest. The exhaustion is all-pervasive, affecting mental clarity and physical ability.",
        "Debilitating fatigue marks chronic fatigue syndrome, with patients experiencing constant exhaustion that rest cannot relieve. Mental fogginess and sleep problems worsen the condition.",
        "Chronic fatigue syndrome causes unrelenting weariness and tiredness that pervades all activities. Patients report feeling exhausted constantly, with poor sleep quality and cognitive difficulties.",
        "Severe, persistent tiredness defines chronic fatigue syndrome. Rest provides no benefit and patients experience profound exhaustion affecting work, social life, and daily tasks.",
        "Chronic fatigue syndrome involves crushing exhaustion that doesn't improve with rest or sleep. Patients struggle with overwhelming tiredness, mental cloudiness, and reduced functioning.",
        "Unrelenting fatigue characterizes chronic fatigue syndrome, with patients experiencing severe exhaustion day after day. Sleep is non-restorative and cognitive abilities decline.",
        "Chronic fatigue syndrome presents as extreme, persistent tiredness unaffected by rest. The exhaustion is all-consuming, preventing normal activities and reducing quality of life.",
        "Overwhelming weariness and exhaustion define chronic fatigue syndrome. Patients experience crushing fatigue that rest doesn't help, along with poor sleep and mental fog.",
        "Chronic fatigue syndrome manifests with profound, unremitting tiredness. Physical and mental exhaustion persist despite adequate rest, severely limiting daily functioning.",
        "Severe exhaustion characterizes chronic fatigue syndrome, where patients feel constantly tired regardless of how much they rest. Cognitive difficulties and poor sleep accompany the fatigue.",
        "Chronic fatigue syndrome involves persistent, overwhelming tiredness that dominates patients' lives. Rest provides no relief and the exhaustion affects both physical and mental capacity.",
        "Relentless exhaustion marks chronic fatigue syndrome, with fatigue that doesn't respond to rest or sleep. Patients experience crushing tiredness and impaired cognitive function.",
        "Chronic fatigue syndrome presents with extreme, unrelenting weariness. The exhaustion is profound and pervasive, affecting all aspects of life despite adequate rest.",
        "Debilitating, persistent fatigue defines chronic fatigue syndrome. Patients struggle with overwhelming tiredness that rest cannot alleviate, plus poor sleep and mental cloudiness.",
        "Chronic fatigue syndrome causes severe, ongoing exhaustion unrelieved by rest. The fatigue is crushing and all-encompassing, dramatically reducing patients' ability to function.",
        "Overwhelming, unremitting tiredness characterizes chronic fatigue syndrome. Despite rest and sleep, the exhaustion persists, severely affecting daily activities and quality of life.",
    ]

    return [(f"chronic_fatigue_variant_{i+1}.txt", templates[i % len(templates)]) for i in range(n)]


def generate_joint_pain_variants(n=25) -> List[Tuple[str, str]]:
    """Generate 25 similar rheumatoid arthritis/joint pain documents."""
    templates = [
        "Rheumatoid arthritis causes painful, swollen joints particularly affecting hands and wrists. Morning stiffness lasting over an hour is characteristic. Joints become warm, tender, and difficult to move.",
        "Joint pain and swelling in hands and wrists mark rheumatoid arthritis. Prolonged morning stiffness is common, with joints feeling tender and warm. Movement becomes painful and limited.",
        "Rheumatoid arthritis presents with tender, swollen joints especially in hands and wrists. Morning stiffness persists for extended periods. Affected joints are warm to touch and painful with movement.",
        "Painful, inflamed joints in hands and wrists characterize rheumatoid arthritis. Extended morning stiffness occurs regularly. Joints are tender, warm, and movement-restricted.",
        "Rheumatoid arthritis affects hands and wrists with painful swelling. Morning stiffness lasts many hours. Joints feel warm, tender, and resist movement.",
        "Swollen, painful joints particularly in hands and wrists indicate rheumatoid arthritis. Prolonged morning stiffness is characteristic. Warmth, tenderness, and limited mobility affect joints.",
        "Rheumatoid arthritis causes joint swelling and pain, especially hands and wrists. Morning stiffness extends beyond one hour. Joints are warm, tender, difficult to move.",
        "Tender, swollen joints in hands and wrists define rheumatoid arthritis. Extended morning stiffness is typical. Affected joints feel warm and movement hurts.",
        "Rheumatoid arthritis manifests with painful hand and wrist joint swelling. Morning stiffness persists extensively. Joints show warmth, tenderness, restricted motion.",
        "Joint swelling and pain in hands and wrists mark rheumatoid arthritis. Prolonged morning stiffness occurs consistently. Warmth, tenderness, and limited movement affect joints.",
        "Rheumatoid arthritis presents painful, swollen hand and wrist joints. Morning stiffness lasts considerable time. Joints are tender, warm, movement-impaired.",
        "Painful hand and wrist swelling characterizes rheumatoid arthritis. Extended morning stiffness is common feature. Joints feel warm, tender, resist movement.",
        "Rheumatoid arthritis involves symmetric joint inflammation primarily in hands and wrists. Prolonged morning stiffness exceeding one hour is diagnostic. Joints are swollen, warm, and tender to touch.",
        "Joint inflammation and pain dominate rheumatoid arthritis, especially affecting small joints of hands. Morning stiffness lasting hours is typical. Swelling, warmth, and tenderness characterize affected joints.",
        "Rheumatoid arthritis causes bilateral joint involvement with pain and swelling. Hands and wrists are commonly affected. Morning stiffness is prolonged and joints feel warm and tender.",
        "Painful swelling of hand and wrist joints marks rheumatoid arthritis. Extended morning stiffness is characteristic. Inflammation causes warmth, tenderness, and reduced mobility.",
        "Rheumatoid arthritis presents with symmetric painful joint swelling. Hands and wrists are primary sites. Prolonged morning stiffness and joint warmth are typical features.",
        "Joint pain, swelling, and stiffness characterize rheumatoid arthritis. Hands and wrists are most affected. Morning stiffness extends beyond an hour with joints tender and warm.",
        "Rheumatoid arthritis affects multiple joints with pain and inflammation. Small joints of hands commonly involved. Morning stiffness is prolonged and joints become swollen and tender.",
        "Painful, tender joints with swelling define rheumatoid arthritis. Hands and wrists show prominent involvement. Extended morning stiffness and joint warmth are diagnostic clues.",
        "Rheumatoid arthritis causes symmetric joint pain and swelling. Morning stiffness lasting over an hour is characteristic. Joints are inflamed, warm, and tender.",
        "Joint inflammation in hands and wrists characterizes rheumatoid arthritis. Prolonged morning stiffness occurs daily. Affected joints are painful, swollen, and warm to touch.",
        "Rheumatoid arthritis presents with bilateral joint swelling and tenderness. Hands and wrists are preferentially affected. Morning stiffness extends for hours.",
        "Painful joint swelling particularly affecting hands and wrists marks rheumatoid arthritis. Morning stiffness is prolonged. Inflammation causes warmth and tenderness.",
        "Rheumatoid arthritis involves symmetric painful swelling of small joints. Hands and wrists commonly affected. Extended morning stiffness and joint warmth are typical.",
    ]

    return [(f"rheumatoid_arthritis_variant_{i+1}.txt", templates[i % len(templates)]) for i in range(n)]


def generate_lupus_variants(n=20) -> List[Tuple[str, str]]:
    """Generate 20 lupus (SLE) documents with overlapping symptoms."""
    templates = [
        "Systemic lupus erythematosus is an autoimmune disease where the immune system attacks healthy tissues. Characteristic butterfly rash across cheeks and nose appears in many patients. Fatigue, joint pain, and fever are common symptoms.",
        "Lupus patients experience diverse symptoms including fatigue, joint pain, and fever. The distinctive malar rash resembles a butterfly pattern on the face. Photosensitivity causes skin reactions to sunlight.",
        "Systemic lupus erythematosus causes chronic fatigue, painful joints, and recurrent fevers. The butterfly-shaped facial rash is characteristic. Immune system dysfunction attacks multiple organs.",
        "Lupus manifests with fatigue, arthritis, and low-grade fevers. Malar rash across the cheeks is distinctive. Sun exposure triggers symptom flares in many patients.",
        "Systemic lupus erythematosus presents with overwhelming fatigue, joint pain, and fever episodes. The characteristic butterfly rash appears on the face. Autoimmune inflammation affects multiple systems.",
        "Lupus patients suffer from persistent fatigue, swollen painful joints, and intermittent fevers. Facial butterfly rash is a hallmark sign. Sunlight sensitivity is common.",
        "Systemic lupus erythematosus causes severe fatigue, arthralgias, and fever. The malar butterfly rash is distinctive. Immune system attacks healthy body tissues.",
        "Lupus symptoms include chronic exhaustion, joint inflammation, and recurrent low-grade fever. Butterfly-pattern facial rash is characteristic. Multiple organs can be affected.",
        "Systemic lupus erythematosus presents with debilitating fatigue, painful swollen joints, and fever. The diagnostic butterfly rash spans the cheeks. Photosensitivity is typical.",
        "Lupus manifests as extreme tiredness, arthritis, and periodic fevers. Malar rash resembling a butterfly marks the face. Autoimmune processes cause widespread inflammation.",
        "Systemic lupus erythematosus causes profound fatigue, joint pain and swelling, plus fever episodes. Butterfly rash across nose and cheeks is diagnostic. Sun triggers flares.",
        "Lupus patients experience unrelenting fatigue, painful joints, and low-grade fevers. The facial butterfly rash is a key feature. Multiple body systems are affected.",
        "Systemic lupus erythematosus involves severe exhaustion, arthritis symptoms, and recurrent fever. Characteristic malar butterfly rash appears on the face. Autoimmune inflammation is systemic.",
        "Lupus presents with crushing fatigue, joint inflammation and pain, and intermittent fever. Butterfly-shaped rash on cheeks and nose is distinctive. Photosensitivity common.",
        "Systemic lupus erythematosus manifests as persistent tiredness, painful swollen joints, and fever. Diagnostic butterfly rash crosses the face. Immune system attacks own tissues.",
        "Lupus symptoms include overwhelming fatigue, arthritis with pain and swelling, and periodic fevers. Malar butterfly rash is characteristic. Sun exposure worsens symptoms.",
        "Systemic lupus erythematosus causes debilitating exhaustion, inflammatory joint pain, and low-grade fever. The butterfly facial rash is a hallmark. Multiple organs affected.",
        "Lupus presents with severe chronic fatigue, painful inflamed joints, and recurrent fever episodes. Butterfly rash spanning cheeks is diagnostic. Autoimmune disease affects many systems.",
        "Systemic lupus erythematosus involves extreme tiredness, arthritis symptoms, and intermittent fever. Characteristic butterfly-pattern rash appears on face. Photosensitivity is typical.",
        "Lupus patients suffer from profound fatigue, joint pain and swelling, and periodic low-grade fever. The malar butterfly rash is distinctive. Immune dysfunction causes widespread problems.",
    ]

    return [(f"lupus_variant_{i+1}.txt", templates[i % len(templates)]) for i in range(n)]


def generate_lyme_variants(n=20) -> List[Tuple[str, str]]:
    """Generate 20 Lyme disease documents."""
    templates = [
        "Lyme disease results from tick-borne Borrelia burgdorferi bacteria. Early symptoms include fever, fatigue, headache, and joint pain. The hallmark erythema migrans rash expands outward in a bull's-eye pattern.",
        "Lyme disease causes fatigue, fever, and joint aches following tick exposure. The characteristic bull's-eye rash appears at the bite site. Untreated infection spreads to joints and nervous system.",
        "Tick-borne Lyme disease presents with fever, extreme fatigue, and joint pain. Expanding circular rash with central clearing is diagnostic. Early treatment prevents complications.",
        "Lyme disease manifests as fever, profound tiredness, and arthralgias. The pathognomonic bull's-eye rash develops at tick bite location. Symptoms worsen without antibiotic therapy.",
        "Borrelia burgdorferi infection causes Lyme disease with fever, fatigue, and joint pain. Erythema migrans bull's-eye rash is characteristic. Tick exposure history aids diagnosis.",
        "Lyme disease presents with low-grade fever, overwhelming fatigue, and painful joints. The expanding target-like rash is distinctive. Early antibiotic treatment is essential.",
        "Tick bite transmits Lyme disease causing fever, severe exhaustion, and arthralgia. Bull's-eye pattern rash appears and expands. Multiple systems affected if untreated.",
        "Lyme disease involves fever, debilitating fatigue, and joint aches. The circular expanding rash with clear center is diagnostic. Tick exposure in endemic areas suggests diagnosis.",
        "Borrelia burgdorferi causes Lyme with fever episodes, extreme tiredness, and joint pain. Erythema migrans target rash is pathognomonic. Early treatment prevents dissemination.",
        "Lyme disease manifests as intermittent fever, crushing fatigue, and arthralgias. The bull's-eye rash expands from tick bite site. Antibiotics cure early infection.",
        "Tick-transmitted Lyme disease causes fever, profound exhaustion, and painful joints. Characteristic circular rash with central clearing appears. Delayed treatment risks complications.",
        "Lyme disease presents with fever, unrelenting fatigue, and joint inflammation. The expanding bull's-eye rash is diagnostic. Endemic areas and tick exposure support diagnosis.",
        "Borrelia infection from tick bite causes fever, severe tiredness, and arthritis symptoms. Erythema migrans rash expands in target pattern. Early antibiotics prevent progression.",
        "Lyme disease involves low-grade fever, debilitating fatigue, and joint pain. The pathognomonic bull's-eye rash develops and enlarges. Tick bite history is important.",
        "Tick-borne Lyme presents with fever episodes, overwhelming exhaustion, and arthralgias. Circular expanding rash with clear center is characteristic. Treatment prevents dissemination.",
        "Lyme disease manifests as fever, extreme fatigue, and painful swollen joints. The diagnostic target-like rash appears at bite site. Outdoor exposure increases risk.",
        "Borrelia burgdorferi infection causes fever, profound tiredness, and joint aches. Bull's-eye erythema migrans rash is distinctive. Early recognition enables effective treatment.",
        "Lyme disease presents with intermittent fever, crushing fatigue, and arthritis. The expanding circular rash with central clearing is diagnostic. Tick exposure history helps.",
        "Tick bite transmits Lyme causing fever, severe exhaustion, and joint inflammation. Characteristic bull's-eye rash expands outward. Antibiotics cure early disease.",
        "Lyme disease involves fever, debilitating fatigue, and painful joints. The pathognomonic expanding target rash appears. Endemic areas and outdoor activities increase risk.",
    ]

    return [(f"lyme_variant_{i+1}.txt", templates[i % len(templates)]) for i in range(n)]


def generate_fibromyalgia_variants(n=20) -> List[Tuple[str, str]]:
    """Generate 20 fibromyalgia documents."""
    templates = [
        "Fibromyalgia causes widespread chronic pain throughout the body and severe fatigue. Specific tender points are sensitive to pressure. Sleep disturbances are common and non-restorative.",
        "Fibromyalgia involves central nervous system pain amplification and exhaustion. The brain processes normal sensations as painful. Widespread tenderness affects muscles with chronic fatigue.",
        "Fibromyalgia presents with diffuse body pain and profound tiredness. Multiple tender points respond painfully to pressure. Sleep quality is poor and unrefreshing.",
        "Fibromyalgia causes chronic widespread musculoskeletal pain and debilitating fatigue. Tender points throughout body are hypersensitive. Sleep architecture is disrupted.",
        "Fibromyalgia manifests as generalized pain affecting all body regions and severe exhaustion. Specific anatomical points are tender to palpation. Rest doesn't relieve fatigue.",
        "Fibromyalgia involves amplified pain processing and chronic tiredness. Widespread muscle pain affects daily functioning. Multiple tender points and poor sleep are characteristic.",
        "Fibromyalgia presents with persistent widespread pain and overwhelming fatigue. Tender point examination reveals hypersensitivity. Sleep is non-restorative and fragmented.",
        "Fibromyalgia causes diffuse chronic pain throughout muscles and profound exhaustion. Specific trigger points are exquisitely tender. Sleep quality is consistently poor.",
        "Fibromyalgia manifests as widespread body pain and debilitating tiredness. Multiple tender points respond painfully to mild pressure. Sleep disturbances worsen symptoms.",
        "Fibromyalgia involves generalized musculoskeletal pain and severe chronic fatigue. Tender points at specific locations are diagnostic. Sleep is unrefreshing.",
        "Fibromyalgia presents with chronic widespread pain and crushing exhaustion. Palpation of tender points elicits pain. Sleep architecture is abnormal.",
        "Fibromyalgia causes diffuse body pain affecting all quadrants and profound fatigue. Specific tender points are hypersensitive. Non-restorative sleep is typical.",
        "Fibromyalgia manifests as persistent widespread musculoskeletal pain and severe tiredness. Multiple anatomical tender points are diagnostic. Sleep quality is poor.",
        "Fibromyalgia involves amplified pain perception and chronic exhaustion. Widespread pain affects muscles and soft tissues. Tender points and sleep problems are characteristic.",
        "Fibromyalgia presents with generalized chronic pain and debilitating fatigue. Specific tender point locations respond painfully to pressure. Sleep is consistently unrefreshing.",
        "Fibromyalgia causes widespread body pain throughout all regions and overwhelming tiredness. Tender point examination reveals hypersensitivity. Sleep disturbances are common.",
        "Fibromyalgia manifests as diffuse chronic musculoskeletal pain and profound exhaustion. Multiple specific tender points are diagnostic. Sleep architecture is disrupted.",
        "Fibromyalgia involves persistent widespread pain and severe chronic fatigue. Tender points at anatomical locations are hypersensitive. Rest doesn't restore energy.",
        "Fibromyalgia presents with generalized body pain and crushing exhaustion. Palpation of specific tender points causes pain. Sleep is non-restorative and fragmented.",
        "Fibromyalgia causes chronic widespread musculoskeletal pain and debilitating tiredness. Multiple tender points respond to light pressure. Sleep quality is consistently poor.",
    ]

    return [(f"fibromyalgia_variant_{i+1}.txt", templates[i % len(templates)]) for i in range(n)]


def generate_hypothyroidism_variants(n=15) -> List[Tuple[str, str]]:
    """Generate 15 hypothyroidism documents."""
    templates = [
        "Hypothyroidism occurs when the thyroid gland produces insufficient hormone. Fatigue, weight gain, and cold intolerance are common. Joint pain and muscle aches frequently occur. Elevated TSH levels confirm diagnosis.",
        "Thyroid hormone deficiency in hypothyroidism causes profound fatigue and weight gain. Patients feel cold easily and experience joint aches. TSH blood test shows elevated levels.",
        "Hypothyroidism manifests with severe tiredness, unexplained weight gain, and cold sensitivity. Joint pain and muscle aches are typical. Low thyroid hormone slows metabolism.",
        "Insufficient thyroid hormone production causes hypothyroidism with overwhelming fatigue. Weight increases despite normal diet and cold intolerance develops. Joint and muscle pain are common.",
        "Hypothyroidism presents with chronic exhaustion, weight gain, and feeling cold constantly. Arthralgias and myalgias affect many patients. TSH elevation indicates deficiency.",
        "Thyroid underactivity in hypothyroidism causes debilitating fatigue and metabolic slowing. Weight gain, cold intolerance, and joint pain are characteristic. Laboratory testing confirms diagnosis.",
        "Hypothyroidism involves extreme tiredness, progressive weight gain, and cold sensitivity. Joint aches and muscle pain frequently occur. Thyroid hormone replacement treats symptoms.",
        "Insufficient thyroid hormone causes hypothyroidism with profound exhaustion. Unexplained weight gain, cold intolerance, and joint pain develop. TSH levels are elevated.",
        "Hypothyroidism manifests as severe chronic fatigue and metabolic dysfunction. Weight increases, cold sensitivity develops, and joints ache. Low thyroid hormone is diagnostic.",
        "Thyroid gland underproduction causes hypothyroidism with overwhelming tiredness. Weight gain occurs despite diet, cold intolerance is prominent, and joint pain is common.",
        "Hypothyroidism presents with debilitating fatigue, unexplained weight increase, and cold sensitivity. Arthralgias affect multiple joints. Elevated TSH confirms diagnosis.",
        "Insufficient thyroid hormone in hypothyroidism causes extreme exhaustion and metabolic slowing. Weight gain, feeling cold, and joint aches are typical features.",
        "Hypothyroidism involves profound fatigue, progressive weight gain despite normal eating, and cold intolerance. Joint and muscle pain are frequent complaints.",
        "Thyroid hormone deficiency causes hypothyroidism with severe tiredness and metabolic changes. Weight increases, cold sensitivity develops, and joints become painful.",
        "Hypothyroidism manifests as crushing fatigue, unexplained weight gain, and constant cold feeling. Joint pain and muscle aches are common. TSH testing is diagnostic.",
    ]

    return [(f"hypothyroidism_variant_{i+1}.txt", templates[i % len(templates)]) for i in range(n)]


def generate_anemia_variants(n=15) -> List[Tuple[str, str]]:
    """Generate 15 anemia documents."""
    templates = [
        "Anemia causes profound fatigue and weakness due to insufficient red blood cells. Patients feel constantly tired and exhausted. Pale skin and shortness of breath are common. Low hemoglobin levels confirm diagnosis.",
        "Iron deficiency anemia presents with severe fatigue and generalized weakness. Exhaustion is overwhelming and constant. Paleness and easy fatigue with exertion are characteristic.",
        "Anemia manifests as extreme tiredness and lack of energy due to reduced oxygen-carrying capacity. Patients experience persistent exhaustion and weakness. Hemoglobin testing reveals deficiency.",
        "Insufficient red blood cells in anemia cause debilitating fatigue. Weakness affects all activities and exhaustion is profound. Pale appearance and laboratory findings are diagnostic.",
        "Anemia presents with crushing fatigue and severe weakness. The tiredness is unrelenting and affects daily functioning. Low red blood cell count reduces oxygen delivery.",
        "Iron deficiency causes anemia with overwhelming exhaustion and weakness. Patients feel tired constantly despite rest. Pale skin and low hemoglobin indicate diagnosis.",
        "Anemia involves severe chronic fatigue due to reduced hemoglobin. Weakness and tiredness dominate symptoms. Blood tests show low red cell count and hemoglobin.",
        "Insufficient oxygen-carrying capacity in anemia causes profound fatigue. Patients experience constant exhaustion and weakness. Paleness and laboratory abnormalities confirm condition.",
        "Anemia manifests with debilitating tiredness and generalized weakness. The fatigue is severe and persistent. Low hemoglobin impairs oxygen delivery to tissues.",
        "Iron deficiency anemia presents with extreme fatigue and overwhelming weakness. Exhaustion affects all activities. Pale appearance and blood tests are diagnostic.",
        "Anemia causes severe chronic fatigue and profound weakness due to low red blood cells. Patients feel constantly tired and exhausted. Hemoglobin levels are reduced.",
        "Reduced oxygen-carrying capacity in anemia leads to crushing fatigue. Weakness and tiredness are constant. Pale skin and laboratory findings indicate diagnosis.",
        "Anemia presents with overwhelming exhaustion and severe weakness. The fatigue is debilitating and unrelenting. Low red cell count and hemoglobin confirm condition.",
        "Iron deficiency causes anemia with profound tiredness and generalized weakness. Patients experience constant fatigue despite rest. Blood testing reveals deficiency.",
        "Anemia manifests as extreme fatigue and persistent weakness due to insufficient hemoglobin. The exhaustion is severe and affects daily life. Laboratory values are diagnostic.",
    ]

    return [(f"anemia_variant_{i+1}.txt", templates[i % len(templates)]) for i in range(n)]


def generate_mononucleosis_variants(n=15) -> List[Tuple[str, str]]:
    """Generate 15 infectious mononucleosis documents."""
    templates = [
        "Infectious mononucleosis causes severe fatigue, fever, and sore throat. Extreme exhaustion lasting weeks is characteristic. Swollen lymph nodes and spleen enlargement occur. Epstein-Barr virus is the cause.",
        "Mononucleosis presents with crushing fatigue, prolonged fever, and pharyngitis. The tiredness is profound and long-lasting. Lymphadenopathy and hepatosplenomegaly are typical.",
        "Epstein-Barr virus causes infectious mononucleosis with debilitating fatigue and fever. Sore throat and swollen glands are common. Exhaustion can persist for months.",
        "Mononucleosis manifests as extreme fatigue, recurrent fever, and throat pain. The exhaustion is overwhelming and prolonged. Enlarged lymph nodes and spleen are characteristic.",
        "Infectious mononucleosis involves severe tiredness, intermittent fever, and pharyngitis. Fatigue lasting weeks to months is typical. Lymph node swelling and splenomegaly occur.",
        "Mononucleosis presents with profound exhaustion, persistent fever, and sore throat. The fatigue is debilitating and long-term. EBV infection causes lymphadenopathy.",
        "Epstein-Barr causes mononucleosis with crushing fatigue and recurrent fever. Throat inflammation and swollen glands are common. Extreme tiredness persists extensively.",
        "Mononucleosis manifests as overwhelming fatigue, prolonged low-grade fever, and pharyngitis. The exhaustion is severe and long-lasting. Spleen and lymph nodes enlarge.",
        "Infectious mononucleosis causes debilitating tiredness, intermittent fever, and throat pain. Fatigue dominates symptoms for weeks. Lymphadenopathy and hepatosplenomegaly are typical.",
        "Mononucleosis presents with extreme exhaustion, persistent fever, and sore throat. The fatigue is profound and prolonged. EBV infection causes characteristic findings.",
        "Epstein-Barr virus causes infectious mono with severe fatigue and recurrent fever. Throat pain and swollen glands occur. The exhaustion can last many weeks.",
        "Mononucleosis manifests as crushing tiredness, prolonged fever episodes, and pharyngitis. The fatigue is overwhelming and persistent. Lymph node and spleen enlargement are common.",
        "Infectious mononucleosis involves profound exhaustion, intermittent fever, and throat inflammation. Fatigue lasting months is characteristic. Hepatosplenomegaly and lymphadenopathy occur.",
        "Mononucleosis presents with debilitating fatigue, persistent low-grade fever, and sore throat. The tiredness is severe and long-term. EBV causes typical symptoms.",
        "Epstein-Barr causes mono with overwhelming exhaustion and recurrent fever. Pharyngitis and swollen lymph nodes are typical. Extreme fatigue persists extensively.",
    ]

    return [(f"mononucleosis_variant_{i+1}.txt", templates[i % len(templates)]) for i in range(n)]


def generate_polymyalgia_variants(n=10) -> List[Tuple[str, str]]:
    """Generate 10 polymyalgia rheumatica documents."""
    templates = [
        "Polymyalgia rheumatica causes severe muscle pain and stiffness in shoulders and hips. Morning stiffness is profound and prolonged. Low-grade fever and fatigue are common. Affects older adults primarily.",
        "Polymyalgia rheumatica presents with bilateral shoulder and hip pain with severe stiffness. Morning symptoms are worst, lasting hours. Fever and exhaustion occur. Elevated inflammatory markers aid diagnosis.",
        "Polymyalgia rheumatica manifests as symmetric muscle pain affecting shoulders and hips. Profound morning stiffness limits movement. Fatigue and low-grade fever are typical. Primarily affects elderly.",
        "Polymyalgia rheumatica involves severe bilateral muscle aching in shoulder and hip girdles. Morning stiffness is extreme and prolonged. Fever episodes and tiredness occur. Age over 50 typical.",
        "Polymyalgia rheumatica causes symmetric pain and stiffness in shoulders and hips. Morning symptoms are debilitating. Fatigue and intermittent fever are common. Inflammatory condition of older adults.",
        "Polymyalgia rheumatica presents with severe bilateral muscle pain affecting proximal joints. Profound morning stiffness lasts hours. Low-grade fever and exhaustion occur. ESR elevation is typical.",
        "Polymyalgia rheumatica manifests as symmetric shoulder and hip muscle pain with extreme stiffness. Mornings are worst with prolonged immobility. Fever and fatigue are characteristic. Elderly patients affected.",
        "Polymyalgia rheumatica involves bilateral pain and profound stiffness in shoulder and hip regions. Morning symptoms severely limit function. Tiredness and fever episodes occur. Inflammatory markers elevated.",
        "Polymyalgia rheumatica causes severe symmetric muscle aching in shoulders and hips. Morning stiffness is extreme and long-lasting. Low-grade fever and fatigue are typical. Age over 50 required.",
        "Polymyalgia rheumatica presents with bilateral proximal muscle pain and debilitating stiffness. Mornings are particularly difficult. Exhaustion and intermittent fever occur. Affects older population primarily.",
    ]

    return [(f"polymyalgia_variant_{i+1}.txt", templates[i % len(templates)]) for i in range(n)]


def generate_depression_variants(n=10) -> List[Tuple[str, str]]:
    """Generate 10 depression documents focusing on fatigue."""
    templates = [
        "Depression causes profound fatigue and lack of energy. Patients feel constantly tired and exhausted despite rest. Motivation decreases and simple tasks become overwhelming. Sleep disturbances are common.",
        "Major depression manifests with severe fatigue and persistent low energy. The tiredness is unrelenting and affects all activities. Loss of interest and hopelessness accompany exhaustion.",
        "Depression presents with crushing fatigue and complete lack of motivation. Patients experience constant exhaustion and difficulty performing tasks. Sleep problems worsen the tiredness.",
        "Major depressive disorder causes debilitating fatigue and energy depletion. The exhaustion is profound and persistent. Daily activities become extremely difficult. Sleep quality suffers.",
        "Depression involves overwhelming fatigue and severe lack of energy. Tiredness affects all aspects of functioning. Motivation disappears and tasks feel impossible. Rest doesn't restore energy.",
        "Major depression manifests as extreme fatigue and profound exhaustion. Energy levels are constantly low. Simple activities become overwhelming challenges. Sleep disturbances contribute to tiredness.",
        "Depression causes severe persistent fatigue and complete energy loss. Patients feel constantly drained and exhausted. Motivation and interest decline. Sleep problems are typical.",
        "Major depressive disorder presents with crushing fatigue and unrelenting tiredness. Energy depletion is profound. Daily functioning becomes severely impaired. Sleep architecture is disrupted.",
        "Depression involves debilitating fatigue and extreme lack of energy. The exhaustion is constant and overwhelming. Tasks require enormous effort. Sleep quality is poor.",
        "Major depression manifests with profound fatigue and persistent exhaustion. Energy levels remain depleted despite rest. Motivation and functioning decline. Sleep disturbances worsen symptoms.",
    ]

    return [(f"depression_variant_{i+1}.txt", templates[i % len(templates)]) for i in range(n)]


def generate_sleep_apnea_variants(n=10) -> List[Tuple[str, str]]:
    """Generate 10 sleep apnea documents."""
    templates = [
        "Obstructive sleep apnea causes severe daytime fatigue due to disrupted nighttime breathing. Patients experience constant exhaustion despite spending time in bed. Loud snoring and witnessed breathing pauses are characteristic.",
        "Sleep apnea presents with overwhelming daytime tiredness from poor quality sleep. Repeated breathing interruptions prevent restful sleep. The fatigue is profound and constant. Snoring is typical.",
        "Obstructive sleep apnea manifests as extreme daytime fatigue and exhaustion. Nighttime breathing cessations fragment sleep. Patients feel constantly tired despite adequate time in bed. Witnessed apneas occur.",
        "Sleep apnea causes debilitating daytime tiredness due to sleep fragmentation. Breathing stops repeatedly during night. The exhaustion is severe and unrelenting. Loud snoring is common.",
        "Obstructive sleep apnea involves crushing daytime fatigue from disrupted sleep architecture. Upper airway obstruction causes breathing pauses. Patients experience constant exhaustion. Bed partner reports snoring.",
        "Sleep apnea presents with profound daytime tiredness and lack of energy. Repeated respiratory events prevent deep sleep. The fatigue is overwhelming and persistent. Snoring and gasping occur.",
        "Obstructive sleep apnea manifests as severe daytime exhaustion despite time in bed. Breathing interruptions fragment sleep cycles. Patients feel constantly drained. Witnessed breathing pauses are diagnostic.",
        "Sleep apnea causes extreme daytime fatigue due to poor sleep quality. Airway collapse during sleep disrupts breathing. The tiredness is debilitating and constant. Loud snoring is typical.",
        "Obstructive sleep apnea involves overwhelming daytime exhaustion from sleep fragmentation. Repeated apneas and hypopneas occur. Patients experience profound fatigue. Bed partner observes breathing pauses.",
        "Sleep apnea presents with debilitating daytime tiredness and lack of refreshment from sleep. Upper airway obstruction causes breathing cessations. The exhaustion is severe. Snoring is characteristic.",
    ]

    return [(f"sleep_apnea_variant_{i+1}.txt", templates[i % len(templates)]) for i in range(n)]


def generate_vitamin_d_variants(n=10) -> List[Tuple[str, str]]:
    """Generate 10 vitamin D deficiency documents."""
    templates = [
        "Vitamin D deficiency causes fatigue, muscle aches, and bone pain. Patients experience persistent tiredness and joint discomfort. Weakness affects daily activities. Low vitamin D levels confirm diagnosis.",
        "Insufficient vitamin D presents with chronic fatigue and musculoskeletal pain. Joint aches and muscle weakness are common. The exhaustion is ongoing. Blood testing reveals deficiency.",
        "Vitamin D deficiency manifests as persistent fatigue, muscle pain, and joint aches. Bone pain may occur. Tiredness affects functioning. Laboratory values show low vitamin D.",
        "Low vitamin D causes fatigue, generalized muscle aches, and joint pain. Patients feel constantly tired and weak. Bone discomfort may develop. Serum testing is diagnostic.",
        "Vitamin D deficiency presents with chronic exhaustion, myalgias, and arthralgias. The fatigue is persistent and affects activities. Muscle weakness is common. Low levels confirm condition.",
        "Insufficient vitamin D manifests as ongoing fatigue and musculoskeletal pain. Joint aches and muscle soreness are typical. The tiredness is constant. Blood work shows deficiency.",
        "Vitamin D deficiency causes persistent fatigue, muscle and joint pain, and weakness. Patients experience ongoing exhaustion. Bone pain may occur. Laboratory testing reveals low levels.",
        "Low vitamin D presents with chronic tiredness, generalized aches, and joint discomfort. Muscle weakness affects function. The fatigue is unrelenting. Serum vitamin D is deficient.",
        "Vitamin D deficiency manifests as constant fatigue, myalgias, and arthralgias. Bone pain and muscle weakness occur. The exhaustion is persistent. Blood testing is diagnostic.",
        "Insufficient vitamin D causes ongoing fatigue and musculoskeletal pain. Joint aches and muscle soreness are characteristic. Patients feel constantly tired. Low serum levels confirm deficiency.",
    ]

    return [(f"vitamin_d_variant_{i+1}.txt", templates[i % len(templates)]) for i in range(n)]


def generate_unique_diverse_docs() -> List[Tuple[str, str]]:
    """Generate unique diverse documents for variety."""
    docs = [
        ("multiple_sclerosis.txt",
         "Multiple sclerosis is an autoimmune disease affecting the central nervous system. Immune cells attack myelin sheath protecting nerve fibers. Symptoms vary based on lesion location: vision problems, weakness, numbness, balance issues. MRI shows characteristic brain and spinal cord lesions. Disease-modifying therapies slow progression."),

        ("diabetes_type1.txt",
         "Type 1 diabetes results from autoimmune destruction of insulin-producing pancreatic beta cells. Blood glucose rises uncontrolled without insulin. Symptoms include excessive thirst, frequent urination, unexplained weight loss, and fatigue. Diagnosis shows elevated blood glucose and often positive autoantibodies. Treatment requires lifelong insulin replacement."),

        ("celiac_disease.txt",
         "Celiac disease is an immune reaction to gluten protein in wheat, barley, and rye. Small intestine lining becomes damaged, impairing nutrient absorption. Symptoms include diarrhea, bloating, weight loss, and fatigue. Diagnosis requires positive antibody tests and intestinal biopsy showing villous atrophy. Strict gluten-free diet allows intestinal healing."),

        ("inflammatory_bowel.txt",
         "Inflammatory bowel disease encompasses Crohn's disease and ulcerative colitis. Chronic inflammation affects gastrointestinal tract causing abdominal pain, diarrhea, and rectal bleeding. Crohn's can involve any part of digestive tract with skip lesions. Ulcerative colitis affects only colon with continuous inflammation. Endoscopy and biopsy establish diagnosis."),

        ("rheumatic_fever.txt",
         "Rheumatic fever follows untreated streptococcal pharyngitis. Immune response damages heart valves, joints, skin, and brain. Migratory polyarthritis affects large joints. Carditis can cause permanent valve damage. Erythema marginatum rash and Sydenham chorea may occur. Jones criteria guide diagnosis."),

        ("sarcoidosis.txt",
         "Sarcoidosis causes granuloma formation in multiple organs, primarily lungs and lymph nodes. Patients present with cough, shortness of breath, and fatigue. Hilar lymphadenopathy on chest X-ray is characteristic. Skin, eyes, heart, and nervous system may be affected. Diagnosis requires biopsy showing non-caseating granulomas."),

        ("parkinsons_disease.txt",
         "Parkinson's disease involves progressive degeneration of dopamine-producing neurons in substantia nigra. Classic triad includes resting tremor, rigidity, and bradykinesia. Postural instability develops later. Symptoms begin asymmetrically. Levodopa therapy improves motor symptoms but doesn't stop progression."),

        ("cushings_syndrome.txt",
         "Cushing's syndrome results from excess cortisol exposure. Central obesity, moon face, buffalo hump, and purple striae develop. Hypertension, diabetes, and osteoporosis occur. Muscle weakness and fatigue are common. Diagnosis requires 24-hour urine cortisol, late-night salivary cortisol, or dexamethasone suppression testing."),

        ("addisons_disease.txt",
         "Addison's disease involves adrenal gland failure to produce sufficient cortisol and aldosterone. Profound fatigue, weakness, weight loss, and hyperpigmentation occur. Salt craving and orthostatic hypotension are typical. Hyponatremia and hyperkalemia on labs. Diagnosis requires ACTH stimulation testing."),

        ("pernicious_anemia.txt",
         "Pernicious anemia results from vitamin B12 deficiency due to intrinsic factor absence. Autoimmune destruction of gastric parietal cells prevents B12 absorption. Fatigue, weakness, and neurological symptoms develop. Macrocytic anemia and elevated MCV on CBC. Anti-intrinsic factor antibodies confirm diagnosis."),
    ]

    return docs


def save_all_documents(output_dir: str):
    """Generate and save all documents."""
    output_path = Path(output_dir)

    # Clear existing files
    if output_path.exists():
        for file in output_path.glob("*.txt"):
            file.unlink()

    output_path.mkdir(parents=True, exist_ok=True)

    # Generate all documents
    all_docs = []
    all_docs.extend(generate_chronic_fatigue_variants(30))
    all_docs.extend(generate_joint_pain_variants(25))
    all_docs.extend(generate_lupus_variants(20))
    all_docs.extend(generate_lyme_variants(20))
    all_docs.extend(generate_fibromyalgia_variants(20))
    all_docs.extend(generate_hypothyroidism_variants(15))
    all_docs.extend(generate_anemia_variants(15))
    all_docs.extend(generate_mononucleosis_variants(15))
    all_docs.extend(generate_polymyalgia_variants(10))
    all_docs.extend(generate_depression_variants(10))
    all_docs.extend(generate_sleep_apnea_variants(10))
    all_docs.extend(generate_vitamin_d_variants(10))
    all_docs.extend(generate_unique_diverse_docs())

    # Save documents
    for filename, content in all_docs:
        filepath = output_path / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

    print(f"Generated {len(all_docs)} documents:")
    print(f"  - 30 chronic fatigue syndrome variants (highly repetitive)")
    print(f"  - 25 rheumatoid arthritis variants (highly repetitive)")
    print(f"  - 20 lupus variants (repetitive, overlapping symptoms)")
    print(f"  - 20 Lyme disease variants (repetitive, overlapping symptoms)")
    print(f"  - 20 fibromyalgia variants (repetitive)")
    print(f"  - 15 hypothyroidism variants (repetitive)")
    print(f"  - 15 anemia variants (repetitive)")
    print(f"  - 15 mononucleosis variants (repetitive)")
    print(f"  - 10 polymyalgia rheumatica variants (repetitive)")
    print(f"  - 10 depression variants (repetitive)")
    print(f"  - 10 sleep apnea variants (repetitive)")
    print(f"  - 10 vitamin D deficiency variants (repetitive)")
    print(f"  - 10 unique diverse disease documents")
    print(f"\nTotal: {len(all_docs)} documents")
    print(f"\nDesign Strategy:")
    print(f"  Query: 'Patient presents with chronic fatigue, joint pain, and occasional low-grade fever'")
    print(f"  - Naive RAG will grab multiple near-duplicate variants from dominant clusters")
    print(f"  - QUBO RAG should recognize redundancy and select diverse disease perspectives")
    print(f"  - MMR will maximize diversity but may sacrifice relevance")
    print(f"  - Large dataset amplifies the demonstration of diversity advantage")


if __name__ == '__main__':
    script_dir = Path(__file__).parent
    save_all_documents(script_dir)
    print(f"\n[OK] Large repetitive dataset generated in {script_dir}")
