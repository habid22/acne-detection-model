"""
Simplified Treatment Recommendation Service for General Acne Detection
"""

from typing import List, Dict, Any

class SimpleTreatmentRecommender:
    """
    Service for recommending treatments based on general acne detection
    """
    
    def __init__(self):
        """Initialize the treatment recommender"""
        self.general_treatments = self._load_general_treatments()
    
    def _load_general_treatments(self) -> Dict[str, Any]:
        """Load general acne treatment information"""
        return {
            "mild": {
                "description": "Mild acne with few lesions",
                "treatments": {
                    "topical": [
                        "Benzoyl peroxide (2.5-5%) - kills bacteria and reduces inflammation",
                        "Salicylic acid (2% BHA) - helps unclog pores and exfoliate",
                        "Gentle cleanser with salicylic acid or benzoyl peroxide"
                    ],
                    "lifestyle": [
                        "Wash face twice daily with gentle cleanser",
                        "Use oil-free, non-comedogenic moisturizer",
                        "Apply daily sunscreen (SPF 30+)",
                        "Avoid picking or squeezing lesions",
                        "Change pillowcases regularly"
                    ],
                    "over_the_counter": [
                        "Acne spot treatments with benzoyl peroxide",
                        "Gentle exfoliating products (2-3 times per week)",
                        "Oil-free moisturizers and sunscreens"
                    ]
                },
                "prevention": [
                    "Maintain consistent skincare routine",
                    "Use non-comedogenic products",
                    "Keep skin clean and moisturized",
                    "Avoid harsh scrubs or over-washing"
                ]
            },
            "moderate": {
                "description": "Moderate acne with multiple lesions",
                "treatments": {
                    "topical": [
                        "Benzoyl peroxide (5-10%) - stronger antibacterial treatment",
                        "Salicylic acid (2% BHA) - pore-clearing exfoliant",
                        "Adapalene (Differin) - over-the-counter retinoid",
                        "Azelaic acid - anti-inflammatory and antibacterial"
                    ],
                    "lifestyle": [
                        "Consistent twice-daily cleansing routine",
                        "Oil-free, non-comedogenic products only",
                        "Daily sunscreen application",
                        "Avoid touching face with hands",
                        "Clean makeup brushes regularly"
                    ],
                    "professional": [
                        "Consider dermatologist consultation",
                        "Professional extractions if needed",
                        "Chemical peels (mild salicylic acid)"
                    ]
                },
                "prevention": [
                    "Early treatment of new lesions",
                    "Stress management techniques",
                    "Balanced diet with limited dairy/sugar",
                    "Regular exercise and adequate sleep"
                ]
            },
            "severe": {
                "description": "Severe acne with many lesions or deep cysts",
                "treatments": {
                    "topical": [
                        "Prescription retinoids (tretinoin, adapalene)",
                        "Prescription benzoyl peroxide combinations",
                        "Antibiotic creams (clindamycin)",
                        "Azelaic acid (prescription strength)"
                    ],
                    "lifestyle": [
                        "Gentle skincare routine - avoid irritation",
                        "Use prescribed medications consistently",
                        "Avoid picking or squeezing lesions",
                        "Apply warm compresses for comfort"
                    ],
                    "professional": [
                        "Dermatologist consultation required",
                        "Oral antibiotics (doxycycline, minocycline)",
                        "Isotretinoin (Accutane) for severe cases",
                        "Cortisone injections for large cysts",
                        "Professional extraction procedures"
                    ]
                },
                "prevention": [
                    "Follow dermatologist's treatment plan",
                    "Monitor for side effects of medications",
                    "Regular follow-up appointments",
                    "Lifestyle modifications as recommended"
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
        severity = detection_results.get("severity", {})
        
        if not detections:
            return {
                "general_advice": [
                    "Maintain a consistent skincare routine",
                    "Use gentle, non-comedogenic products",
                    "Apply daily sunscreen",
                    "Keep skin clean and moisturized"
                ],
                "message": "No acne detected. Continue with preventive care.",
                "severity_level": "none"
            }
        
        # Get severity level
        severity_level = severity.get("level", "mild")
        total_detections = len(detections)
        
        # Get recommendations based on severity
        if severity_level in self.general_treatments:
            treatment_info = self.general_treatments[severity_level]
        else:
            # Fallback to mild if severity not recognized
            treatment_info = self.general_treatments["mild"]
            severity_level = "mild"
        
        # Calculate confidence-based adjustments
        avg_confidence = sum(d["confidence"] for d in detections) / len(detections)
        
        # Adjust recommendations based on confidence
        if avg_confidence < 0.6:
            confidence_note = "Note: Detection confidence is moderate. Consider professional evaluation."
        elif avg_confidence > 0.8:
            confidence_note = "High confidence detection. Treatment recommendations are reliable."
        else:
            confidence_note = "Good detection confidence. Follow recommendations consistently."
        
        return {
            "severity_level": severity_level,
            "total_detections": total_detections,
            "average_confidence": avg_confidence,
            "confidence_note": confidence_note,
            "description": treatment_info["description"],
            "treatments": treatment_info["treatments"],
            "prevention": treatment_info["prevention"],
            "message": f"Detected {total_detections} acne lesions. Severity: {severity_level.title()}",
            "next_steps": self._get_next_steps(severity_level, total_detections)
        }
    
    def _get_next_steps(self, severity_level: str, total_detections: int) -> List[str]:
        """Get next steps based on severity and detection count"""
        if severity_level == "severe" or total_detections > 20:
            return [
                "Schedule dermatologist appointment within 1-2 weeks",
                "Start gentle skincare routine immediately",
                "Avoid picking or squeezing lesions",
                "Consider over-the-counter treatments while waiting for appointment"
            ]
        elif severity_level == "moderate" or total_detections > 10:
            return [
                "Start treatment with over-the-counter products",
                "Monitor progress for 4-6 weeks",
                "Consider dermatologist if no improvement",
                "Maintain consistent skincare routine"
            ]
        else:
            return [
                "Start with gentle over-the-counter treatments",
                "Monitor for 2-4 weeks",
                "Continue preventive care",
                "See dermatologist if condition worsens"
            ]
    
    def get_emergency_advice(self) -> List[str]:
        """Get emergency advice for severe cases"""
        return [
            "If you have severe, painful acne with deep cysts, consult a dermatologist immediately",
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
    
    def get_all_treatments(self) -> Dict[str, Any]:
        """Get all available treatment information"""
        return self.general_treatments
