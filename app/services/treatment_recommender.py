"""
Treatment Recommendation Service
"""

from typing import List, Dict, Any
import json
from pathlib import Path

class TreatmentRecommender:
    """
    Service for recommending treatments based on detected acne types and severity
    """
    
    def __init__(self):
        """Initialize the treatment recommender with knowledge base"""
        self.treatment_knowledge = self._load_treatment_knowledge()
    
    def _load_treatment_knowledge(self) -> Dict[str, Any]:
        """
        Load treatment knowledge base
        
        Returns:
            Dictionary containing treatment information
        """
        return {
            "blackheads": {
                "description": "Open comedones caused by clogged pores",
                "severity": "mild",
                "treatments": {
                    "topical": [
                        "Salicylic acid (2% BHA) - helps unclog pores",
                        "Benzoyl peroxide (2.5-5%) - kills bacteria and reduces inflammation",
                        "Retinoids (adapalene) - promotes cell turnover"
                    ],
                    "lifestyle": [
                        "Gentle cleansing twice daily",
                        "Non-comedogenic moisturizer",
                        "Regular exfoliation (2-3 times per week)",
                        "Avoid picking or squeezing"
                    ],
                    "professional": [
                        "Extraction by dermatologist",
                        "Chemical peels (salicylic acid)",
                        "Microdermabrasion"
                    ]
                },
                "prevention": [
                    "Use oil-free, non-comedogenic products",
                    "Remove makeup before bed",
                    "Keep hair clean and away from face",
                    "Change pillowcases regularly"
                ]
            },
            "whiteheads": {
                "description": "Closed comedones - small white bumps under the skin",
                "severity": "mild",
                "treatments": {
                    "topical": [
                        "Salicylic acid (2% BHA) - exfoliates and unclogs pores",
                        "Benzoyl peroxide (2.5-5%) - antibacterial and anti-inflammatory",
                        "Retinoids (tretinoin, adapalene) - increases cell turnover"
                    ],
                    "lifestyle": [
                        "Gentle, non-abrasive cleansing",
                        "Oil-free moisturizer",
                        "Avoid harsh scrubs",
                        "Don't pick or squeeze"
                    ],
                    "professional": [
                        "Professional extraction",
                        "Chemical peels",
                        "Microdermabrasion"
                    ]
                },
                "prevention": [
                    "Consistent skincare routine",
                    "Non-comedogenic products",
                    "Regular gentle exfoliation",
                    "Proper hydration"
                ]
            },
            "papules": {
                "description": "Small, raised, inflamed bumps without pus",
                "severity": "moderate",
                "treatments": {
                    "topical": [
                        "Benzoyl peroxide (5-10%) - reduces inflammation",
                        "Salicylic acid (2% BHA) - unclogs pores",
                        "Retinoids - prevents new lesions",
                        "Azelaic acid (15-20%) - anti-inflammatory"
                    ],
                    "lifestyle": [
                        "Gentle cleansing with warm water",
                        "Oil-free, non-comedogenic moisturizer",
                        "Avoid touching face",
                        "Use clean towels and pillowcases"
                    ],
                    "professional": [
                        "Prescription retinoids",
                        "Oral antibiotics (if severe)",
                        "Chemical peels",
                        "Light therapy"
                    ]
                },
                "prevention": [
                    "Consistent anti-acne routine",
                    "Stress management",
                    "Balanced diet",
                    "Regular exercise"
                ]
            },
            "pustules": {
                "description": "Inflamed lesions filled with pus",
                "severity": "moderate",
                "treatments": {
                    "topical": [
                        "Benzoyl peroxide (5-10%) - kills bacteria",
                        "Salicylic acid (2% BHA) - exfoliates",
                        "Retinoids - prevents new lesions",
                        "Antibiotic creams (clindamycin)"
                    ],
                    "lifestyle": [
                        "Gentle cleansing twice daily",
                        "Don't pop or squeeze",
                        "Use clean hands and tools",
                        "Apply warm compress for comfort"
                    ],
                    "professional": [
                        "Prescription antibiotics",
                        "Professional extraction",
                        "Chemical peels",
                        "Light therapy"
                    ]
                },
                "prevention": [
                    "Maintain clean skin",
                    "Avoid picking",
                    "Use antibacterial products",
                    "Manage stress"
                ]
            },
            "nodules": {
                "description": "Large, painful, deep lesions under the skin",
                "severity": "severe",
                "treatments": {
                    "topical": [
                        "Prescription retinoids (tretinoin)",
                        "Benzoyl peroxide (10%)",
                        "Antibiotic creams",
                        "Azelaic acid (20%)"
                    ],
                    "lifestyle": [
                        "Gentle skin care",
                        "Avoid picking or squeezing",
                        "Apply warm compress",
                        "Use non-comedogenic products"
                    ],
                    "professional": [
                        "Oral antibiotics (doxycycline, minocycline)",
                        "Isotretinoin (Accutane) - for severe cases",
                        "Cortisone injections",
                        "Professional extraction",
                        "Chemical peels"
                    ]
                },
                "prevention": [
                    "Early treatment of mild acne",
                    "Consistent skincare routine",
                    "Professional monitoring",
                    "Lifestyle modifications"
                ]
            },
            "dark_spots": {
                "description": "Post-inflammatory hyperpigmentation from previous acne",
                "severity": "mild",
                "treatments": {
                    "topical": [
                        "Vitamin C serum - brightens skin",
                        "Niacinamide (5-10%) - reduces pigmentation",
                        "Alpha arbutin - lightens dark spots",
                        "Retinoids - promotes cell turnover",
                        "Hydroquinone (2-4%) - prescription strength"
                    ],
                    "lifestyle": [
                        "Daily SPF 30+ sunscreen",
                        "Gentle exfoliation",
                        "Consistent skincare routine",
                        "Avoid picking at acne"
                    ],
                    "professional": [
                        "Chemical peels (glycolic acid)",
                        "Laser therapy",
                        "Microdermabrasion",
                        "Prescription hydroquinone"
                    ]
                },
                "prevention": [
                    "Prevent acne formation",
                    "Daily sunscreen use",
                    "Don't pick at acne",
                    "Early treatment of inflammation"
                ]
            }
        }
    
    def get_recommendations(self, detection_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get treatment recommendations based on detection results
        
        Args:
            detection_results: Results from acne detection
            
        Returns:
            Treatment recommendations
        """
        detections = detection_results.get("detections", [])
        
        if not detections:
            return {
                "general_advice": [
                    "Maintain a consistent skincare routine",
                    "Use gentle, non-comedogenic products",
                    "Apply daily sunscreen",
                    "Keep skin clean and moisturized"
                ],
                "message": "No specific acne detected. Continue with preventive care."
            }
        
        # Analyze detected acne types
        detected_classes = set()
        for detection in detections:
            detected_classes.add(detection["class"])
        
        # Get recommendations for each detected type
        recommendations = {}
        for acne_type in detected_classes:
            if acne_type in self.treatment_knowledge:
                recommendations[acne_type] = self.treatment_knowledge[acne_type]
        
        # Determine overall severity and priority
        severity_levels = [self.treatment_knowledge.get(acne_type, {}).get("severity", "mild") 
                          for acne_type in detected_classes]
        
        if "severe" in severity_levels:
            priority = "high"
            general_advice = [
                "Consult a dermatologist for severe acne",
                "Consider prescription treatments",
                "Avoid picking or squeezing lesions",
                "Maintain gentle skincare routine"
            ]
        elif "moderate" in severity_levels:
            priority = "medium"
            general_advice = [
                "Use over-the-counter treatments consistently",
                "Consider professional consultation if no improvement",
                "Maintain good skincare habits",
                "Be patient - results take 4-8 weeks"
            ]
        else:
            priority = "low"
            general_advice = [
                "Continue with gentle skincare routine",
                "Use mild treatments as needed",
                "Monitor for any changes",
                "Maintain preventive care"
            ]
        
        return {
            "recommendations": recommendations,
            "priority": priority,
            "general_advice": general_advice,
            "detected_types": list(detected_classes),
            "message": f"Treatment recommendations for {len(detected_classes)} acne type(s) detected"
        }
    
    def get_all_treatments(self) -> Dict[str, Any]:
        """
        Get all available treatment information
        
        Returns:
            Complete treatment knowledge base
        """
        return self.treatment_knowledge
    
    def get_treatment_by_type(self, acne_type: str) -> Dict[str, Any]:
        """
        Get treatment information for a specific acne type
        
        Args:
            acne_type: Type of acne
            
        Returns:
            Treatment information for the specified type
        """
        return self.treatment_knowledge.get(acne_type, {})
    
    def get_emergency_advice(self) -> List[str]:
        """
        Get emergency advice for severe acne cases
        
        Returns:
            List of emergency advice
        """
        return [
            "If you have severe, painful acne with nodules or cysts, consult a dermatologist immediately",
            "Signs that require immediate medical attention:",
            "- Large, painful lesions that don't respond to treatment",
            "- Acne that's causing significant emotional distress",
            "- Signs of infection (increased redness, warmth, pus)",
            "- Acne that's leaving deep scars",
            "In the meantime:",
            "- Don't pick or squeeze lesions",
            "- Use gentle, non-irritating products",
            "- Apply warm compresses for comfort",
            "- Keep skin clean and moisturized"
        ]
