# Ethics and Safeguards

This project investigates pedagogy bias while maintaining a focus on participant safety and privacy.

## Consent and participation
- Ensure institutional approval before red-teaming live educational data.
- Obtain informed consent from guardians or data owners before collecting new EHCP material.
- Document provenance for all datasets and respect local retention laws.

## Handling sensitive data
- Redact student-identifiable information before storing or sharing artefacts.
- Store secret keys only in environment variables or secure vaults.
- Do not upload raw EHCP profiles to public issue trackers or cloud drives without approval.

## Hot prompt variants
- Hot variants may intentionally push model boundaries and should default to disabled.
- Require the `--ack-hot` flag or equivalent acknowledgement before enabling hot runs.
- Log usage of hot variants for later audit.

## Responsible reporting
- Present quantitative findings with context and note limitations.
- Avoid framing differences as immutable characteristics of student groups.
- Share reproducible configurations so results can be validated independently.
