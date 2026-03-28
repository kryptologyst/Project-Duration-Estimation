"""
Compliance and governance features for project duration estimation.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path


@dataclass
class ComplianceRecord:
    """Record of compliance checks and governance actions."""
    timestamp: datetime
    check_type: str
    status: str
    details: Dict[str, Any]
    user_id: Optional[str] = None
    session_id: Optional[str] = None


@dataclass
class DataLineage:
    """Data lineage tracking for audit purposes."""
    data_source: str
    processing_steps: List[str]
    transformations: List[str]
    output_files: List[str]
    timestamp: datetime
    version: str


class ComplianceManager:
    """Manages compliance, governance, and audit features."""
    
    def __init__(self, audit_log_path: str = "logs/audit.log"):
        """
        Initialize compliance manager.
        
        Args:
            audit_log_path: Path to audit log file
        """
        self.audit_log_path = Path(audit_log_path)
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.compliance_records: List[ComplianceRecord] = []
        self.data_lineage: List[DataLineage] = []
    
    def log_data_access(self, 
                       data_source: str, 
                       user_id: Optional[str] = None,
                       session_id: Optional[str] = None) -> None:
        """
        Log data access for audit purposes.
        
        Args:
            data_source: Source of data being accessed
            user_id: Optional user identifier
            session_id: Optional session identifier
        """
        record = ComplianceRecord(
            timestamp=datetime.now(),
            check_type="data_access",
            status="success",
            details={"data_source": data_source},
            user_id=user_id,
            session_id=session_id
        )
        self.compliance_records.append(record)
        self._write_audit_log(record)
    
    def log_model_execution(self, 
                          model_name: str,
                          input_data_hash: str,
                          output_data_hash: str,
                          execution_time: float,
                          user_id: Optional[str] = None) -> None:
        """
        Log model execution for audit purposes.
        
        Args:
            model_name: Name of the model executed
            input_data_hash: Hash of input data
            output_data_hash: Hash of output data
            execution_time: Time taken for execution
            user_id: Optional user identifier
        """
        record = ComplianceRecord(
            timestamp=datetime.now(),
            check_type="model_execution",
            status="success",
            details={
                "model_name": model_name,
                "input_data_hash": input_data_hash,
                "output_data_hash": output_data_hash,
                "execution_time": execution_time
            },
            user_id=user_id
        )
        self.compliance_records.append(record)
        self._write_audit_log(record)
    
    def log_decision_support(self, 
                           decision_type: str,
                           recommendations: List[str],
                           confidence_scores: Dict[str, float],
                           user_id: Optional[str] = None) -> None:
        """
        Log decision support activities.
        
        Args:
            decision_type: Type of decision being supported
            recommendations: List of recommendations provided
            confidence_scores: Confidence scores for recommendations
            user_id: Optional user identifier
        """
        record = ComplianceRecord(
            timestamp=datetime.now(),
            check_type="decision_support",
            status="success",
            details={
                "decision_type": decision_type,
                "recommendations": recommendations,
                "confidence_scores": confidence_scores,
                "human_review_required": True
            },
            user_id=user_id
        )
        self.compliance_records.append(record)
        self._write_audit_log(record)
    
    def track_data_lineage(self, 
                          data_source: str,
                          processing_steps: List[str],
                          transformations: List[str],
                          output_files: List[str],
                          version: str = "1.0") -> None:
        """
        Track data lineage for audit purposes.
        
        Args:
            data_source: Source of the data
            processing_steps: Steps taken to process the data
            transformations: Transformations applied to the data
            output_files: Output files generated
            version: Version of the processing pipeline
        """
        lineage = DataLineage(
            data_source=data_source,
            processing_steps=processing_steps,
            transformations=transformations,
            output_files=output_files,
            timestamp=datetime.now(),
            version=version
        )
        self.data_lineage.append(lineage)
    
    def validate_data_privacy(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate data for privacy compliance.
        
        Args:
            data: Data to validate
            
        Returns:
            Validation results
        """
        validation_results = {
            "is_compliant": True,
            "issues": [],
            "recommendations": []
        }
        
        # Check for potential PII
        pii_keywords = ["name", "email", "phone", "address", "ssn", "id"]
        for key, value in data.items():
            if any(keyword in key.lower() for keyword in pii_keywords):
                validation_results["issues"].append(f"Potential PII detected in field: {key}")
                validation_results["recommendations"].append(f"Consider anonymizing or hashing field: {key}")
        
        if validation_results["issues"]:
            validation_results["is_compliant"] = False
        
        # Log validation
        record = ComplianceRecord(
            timestamp=datetime.now(),
            check_type="privacy_validation",
            status="success" if validation_results["is_compliant"] else "warning",
            details=validation_results
        )
        self.compliance_records.append(record)
        self._write_audit_log(record)
        
        return validation_results
    
    def generate_compliance_report(self) -> Dict[str, Any]:
        """
        Generate compliance report.
        
        Returns:
            Compliance report
        """
        report = {
            "generated_at": datetime.now().isoformat(),
            "total_records": len(self.compliance_records),
            "data_lineage_entries": len(self.data_lineage),
            "compliance_summary": self._summarize_compliance(),
            "audit_trail": self._generate_audit_trail(),
            "data_lineage": self._generate_lineage_report()
        }
        
        return report
    
    def _summarize_compliance(self) -> Dict[str, Any]:
        """Generate compliance summary."""
        check_types = {}
        status_counts = {"success": 0, "warning": 0, "error": 0}
        
        for record in self.compliance_records:
            check_types[record.check_type] = check_types.get(record.check_type, 0) + 1
            status_counts[record.status] = status_counts.get(record.status, 0) + 1
        
        return {
            "check_types": check_types,
            "status_counts": status_counts,
            "compliance_score": status_counts["success"] / len(self.compliance_records) if self.compliance_records else 0
        }
    
    def _generate_audit_trail(self) -> List[Dict[str, Any]]:
        """Generate audit trail."""
        return [
            {
                "timestamp": record.timestamp.isoformat(),
                "check_type": record.check_type,
                "status": record.status,
                "user_id": record.user_id,
                "session_id": record.session_id,
                "details": record.details
            }
            for record in self.compliance_records
        ]
    
    def _generate_lineage_report(self) -> List[Dict[str, Any]]:
        """Generate data lineage report."""
        return [
            {
                "data_source": lineage.data_source,
                "processing_steps": lineage.processing_steps,
                "transformations": lineage.transformations,
                "output_files": lineage.output_files,
                "timestamp": lineage.timestamp.isoformat(),
                "version": lineage.version
            }
            for lineage in self.data_lineage
        ]
    
    def _write_audit_log(self, record: ComplianceRecord) -> None:
        """Write audit log entry to file."""
        log_entry = {
            "timestamp": record.timestamp.isoformat(),
            "check_type": record.check_type,
            "status": record.status,
            "user_id": record.user_id,
            "session_id": record.session_id,
            "details": record.details
        }
        
        with open(self.audit_log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def export_audit_log(self, output_path: str) -> None:
        """
        Export audit log to file.
        
        Args:
            output_path: Path to export audit log
        """
        report = self.generate_compliance_report()
        
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)


def create_disclaimer_text() -> str:
    """
    Create disclaimer text for the application.
    
    Returns:
        Disclaimer text
    """
    return """
IMPORTANT DISCLAIMER

This is an experimental research and educational tool for project duration estimation. 
It is NOT intended for automated decision-making without human review.

Key Points:
- All estimates are based on mathematical models and historical patterns
- Results should be validated by experienced project managers
- External factors not captured in the model may significantly impact actual project duration
- This tool does not account for resource constraints, stakeholder delays, or unforeseen circumstances
- Use at your own risk and always consult with qualified professionals

The developers and contributors to this tool:
- Make no warranties about the accuracy of estimates
- Are not responsible for any decisions made based on these estimates
- Recommend human oversight for all project planning decisions
- Encourage validation against real-world project data

This tool is provided for educational and research purposes only.
"""


def create_privacy_policy() -> str:
    """
    Create privacy policy text.
    
    Returns:
        Privacy policy text
    """
    return """
PRIVACY POLICY

Data Handling:
- This tool processes project data locally on your system
- No data is transmitted to external servers without explicit consent
- All generated data remains under your control
- Synthetic data generation uses deterministic algorithms for reproducibility

Data Retention:
- Audit logs are maintained locally for compliance purposes
- You can delete all generated data at any time
- No personal information is collected or stored

Compliance:
- This tool follows data minimization principles
- All data processing is transparent and auditable
- Users maintain full control over their data
- No automated decision-making without human oversight

For questions about data handling, please review the source code or contact the development team.
"""
