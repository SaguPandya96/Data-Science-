# Responsible AI and use policy

> This system provides a statistical estimate and should not be treated as
> proof of authorship.

This policy applies to the local interface, CLI, API, saved evaluation results,
and any downstream presentation of AuthentiText output. The version 1 lexical
model is research software with measured failures. It is not approved for
production or consequential decision-making.

## What an output means

AuthentiText returns `likely_human`, `uncertain`, or `likely_machine` under one
frozen, validation-selected policy. These are evidence categories, not facts
about who wrote a passage. The calibrated machine likelihood estimates target
frequency under the MAGE validation setup; it is not a universal authorship
probability. An `uncertain` result means the score falls between the two frozen
thresholds. A decisive result does not mean certainty.

The model does not identify ChatGPT, another specific generator, a person, or a
writing process. It cannot determine intent, originality, plagiarism,
misconduct, factuality, or quality.

## False-positive and false-negative risk

A human false-machine result can lead to an unjust accusation. A machine
false-human result can create false reassurance. Both error types occurred in
the untouched in-distribution test, and error rates varied sharply across
domains and text lengths. The abstention interval reduced forced decisions but
did not eliminate either error.

Aggregate metrics do not establish reliability for an individual, institution,
domain, or future input. The exact frozen results, subgroup failures, Wilson
intervals, and development OOD degradation are in the
[model card](MODEL_CARD.md). They must accompany any claim about model
performance; a favorable aggregate score must not be reported without the
human false-machine, machine false-human, uncertainty, and shift results.

## Distribution shift

The first cycle was developed on English MAGE data. Vocabulary, topic, source,
formatting, generator, and benchmark artifacts may drive the lexical score.
Behavior is unmeasured or weakly measured for:

- languages other than English;
- domains outside the acquired MAGE sources;
- newly released or unseen generators;
- mixed human and machine authorship;
- quoted or copied passages, code, tables, and reference lists;
- heavy human editing, most adversarial edits, and unusual formatting; and
- most cross-dataset use beyond the three English Ghostbuster domains and its
  older ChatGPT/Claude conditions.

The MAGE GPT-4 and paraphrase files are development stress tests, not an
external validation. Their degraded results show that a calibrated and
abstaining policy can still fail under shift. Drift signals exposed by the
local service are investigation prompts only; they do not establish cause,
repair the score, or authorize automatic retraining.

The sealed Ghostbuster evaluation is real external evidence, not a production
approval. Its frozen policy called 12.5334% of human records likely machine,
including 24.7485% of human student essays, while remaining uncertain on
42.9660% of all records. Those outcomes make educational accusations especially
unsafe. Its English-only, three-domain, older-generator corpus does not validate
new domains, languages, or generators.

## Short, long, and mixed text

Text below 50 whitespace tokens receives a low-evidence warning because the
frozen short-text slice performed poorly. The category is not overridden, so a
reviewer must not ignore that warning. Inputs outside the observed development
length range also receive a warning. The current lexical model accepts the
complete validated input; it does not silently truncate, segment authors, or
locate which sentences came from which source.

Mixed-authorship detection was not trained or evaluated. Combining human and
machine passages can produce a score with no supported interpretation. Segment
scores, if added in a future version, must be described as detector scores and
not proof about individual sentences.

## Required human review

AuthentiText may be used only as one weak, contestable input to an exploratory
review. A reviewer should:

1. read the full text and all warnings rather than only the category or score;
2. establish whether the language, domain, length, and writing process are
   represented by evaluated data;
3. seek independent, context-appropriate evidence and consider benign
   explanations such as templates, accessibility tools, translation, editing,
   or formulaic writing;
4. document uncertainty and avoid converting a score into an attribution; and
5. conclude that the detector is inconclusive whenever supported evidence is
   insufficient.

No person should be compelled to prove authorship because of an AuthentiText
score. Any process that nevertheless affects a person must provide notice,
meaningful human review, access to the evidence, and a way to contest and
correct the record. Those safeguards do not make the current model suitable
for consequential use; such use remains out of scope.

## Prohibited and out-of-scope uses

Do not use AuthentiText as the sole or primary basis for:

- academic discipline, grading penalties, admissions, or accusations of
  cheating;
- employment, insurance, credit, immigration, legal, or benefits decisions;
- fraud or plagiarism findings;
- account suspension, content removal, or other moderation sanctions;
- law-enforcement or intelligence decisions; or
- bulk surveillance, profiling, or ranking of people.

Do not market the output as “AI detected,” proof, ground truth, or attribution
to a named model. Do not hide the uncertain state, warnings, model version, or
known failure evidence. Do not use the project documentation to provide
instructions for evading detectors.

## Privacy

Submitted text can be sensitive or identifying. Version 1 does not persist raw
text, excerpts, tokens, per-request records, or text hashes. Successful logs
contain only category and coarse input counts; process-local monitoring retains
bounded latency samples and aggregate distributions. There is no database,
remote telemetry, analytics script, cookie, or browser-storage path.

This boundary covers the application, not the operator's shell history, input
files, browser, reverse proxy, host logs, backups, or screen capture. A future
deployment, persistence feature, feedback mechanism, or remote monitoring
export requires a new data-flow review, explicit retention and deletion rules,
access controls, and documentation before collection begins. See the
[inference contract](inference/contract.md),
[API privacy boundary](api/service.md), and
[monitoring contract](operations/monitoring.md).

## Misuse and incident response

Foreseeable misuse includes automated accusations, laundering a score into an
authoritative-sounding claim, selectively reporting favorable metrics, mass
collection of submitted writing, and tuning against the detector to mislead
others. Interface wording and abstention reduce none of these risks by
themselves.

If the system produces a harmful or materially misleading pattern:

1. stop the affected use and do not lower thresholds or suppress warnings;
2. preserve only privacy-safe incident metadata and the exact model/report
   identities;
3. determine whether the input is out of scope, a known subgroup failure, an
   implementation fault, or a new shift;
4. evaluate a proposed change on predefined, provenance-checked data; and
5. treat any replacement as a new model cycle with calibration, human
   false-positive, OOD, and acceptance-gate review.

Drift or one improved metric must never trigger automatic retraining or model
promotion. The [model-selection record](MODEL_SELECTION.md) defines the minimum
gate for a future candidate, and the
[retraining design](operations/retraining.md) defines the complete evidence,
promotion, and rollback sequence.

## Current governance gaps

There has been no multilingual evaluation, real-user study, formal red-team
exercise, accessibility review with assistive technology, or production
privacy assessment. The external corpus and technical deployment acceptance
do not establish production safety. No
formal appeal, incident-owner, or release-approval process exists. These are
unresolved requirements, not features implied by this document.
