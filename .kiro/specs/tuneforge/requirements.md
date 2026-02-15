# Requirements Document: TuneForge

## Introduction

TuneForge is a no-code platform that automates end-to-end LLM fine-tuning pipeline creation, focusing on high-quality instruction dataset generation from raw documents. The platform enables users without deep ML expertise to customize language models for domain-specific tasks through an intuitive interface that handles document ingestion, intelligent chunking, instruction generation, dataset creation, LoRA-based fine-tuning, and model comparison.

## Glossary

- **System**: The TuneForge platform
- **User**: A person interacting with the platform (AI consultant, technical founder, or advanced user)
- **Document**: A file uploaded by the user (PDF, TXT, or DOCX format)
- **Chunk**: A semantically meaningful segment of text extracted from a document
- **Intent**: The purpose or task type for which the model will be fine-tuned (e.g., Q&A, summarization, classification)
- **Instruction**: A prompt or query that forms the input part of a training example
- **Response**: The expected output corresponding to an instruction in a training example
- **Training_Example**: A paired instruction-response data point used for fine-tuning
- **Dataset**: A collection of training examples formatted for model training
- **Base_Model**: The pre-trained language model before fine-tuning
- **Tuned_Model**: The customized model after LoRA fine-tuning
- **LoRA**: Low-Rank Adaptation, a parameter-efficient fine-tuning technique
- **Adapter**: The trained LoRA weights that modify the base model's behavior
- **JSONL**: JSON Lines format, where each line is a valid JSON object

## Requirements

### Requirement 1: Document Upload and Ingestion

**User Story:** As a user, I want to upload documents in common formats, so that I can use my existing content as the foundation for fine-tuning data.

#### Acceptance Criteria

1. WHEN a user uploads a PDF file, THE System SHALL extract all text content from the file
2. WHEN a user uploads a TXT file, THE System SHALL read the text content with proper encoding detection
3. WHEN a user uploads a DOCX file, THE System SHALL extract text content preserving paragraph structure
4. WHEN a user uploads a file exceeding 50MB, THE System SHALL reject the upload and display a size limit error
5. WHEN a user uploads a file with an unsupported format, THE System SHALL reject the upload and display a format error
6. WHEN text extraction fails, THE System SHALL log the error and notify the user with a descriptive message
7. THE System SHALL support uploading multiple documents in a single session

### Requirement 2: Intelligent Text Chunking

**User Story:** As a user, I want the system to automatically break documents into meaningful segments, so that training examples are coherent and contextually complete.

#### Acceptance Criteria

1. WHEN a document is processed, THE System SHALL split the text into chunks based on semantic boundaries
2. WHEN creating chunks, THE System SHALL maintain a target size between 200 and 1000 tokens
3. WHEN a chunk would exceed the maximum size, THE System SHALL split at the nearest sentence boundary
4. WHEN a chunk would be below the minimum size, THE System SHALL merge it with adjacent chunks
5. THE System SHALL preserve context by including overlapping content between consecutive chunks
6. WHEN chunking is complete, THE System SHALL store all chunks with their source document metadata

### Requirement 3: Intent Selection

**User Story:** As a user, I want to specify what task my model should perform, so that the system generates appropriate training data for my use case.

#### Acceptance Criteria

1. THE System SHALL provide a selection interface with predefined intent options
2. THE System SHALL support the following intents: question-answering, summarization, tone-rewriting, classification, and general-assistant
3. WHEN a user selects an intent, THE System SHALL store the selection for dataset generation
4. WHEN a user changes the selected intent, THE System SHALL update the generation strategy accordingly
5. THE System SHALL display a description for each intent option to guide user selection

### Requirement 4: Instruction-Response Generation

**User Story:** As a user, I want the system to automatically create training examples from my documents, so that I don't have to manually write hundreds of instruction-response pairs.

#### Acceptance Criteria

1. WHEN a user initiates generation, THE System SHALL create training examples based on the selected intent and document chunks
2. WHEN the intent is question-answering, THE System SHALL generate questions that can be answered using the chunk content
3. WHEN the intent is summarization, THE System SHALL generate instructions requesting summaries and create corresponding summary responses
4. WHEN the intent is tone-rewriting, THE System SHALL generate instructions for style transformation and create rewritten versions
5. WHEN the intent is classification, THE System SHALL generate classification instructions and appropriate category labels
6. WHEN the intent is general-assistant, THE System SHALL generate diverse conversational instructions with helpful responses
7. THE System SHALL use a language model to generate high-quality instruction-response pairs
8. WHEN generation fails for a chunk, THE System SHALL log the failure and continue processing remaining chunks
9. THE System SHALL generate at least one training example per chunk when possible

### Requirement 5: Dataset Review and Editing

**User Story:** As a user, I want to review and modify generated training examples, so that I can ensure quality and correctness before fine-tuning.

#### Acceptance Criteria

1. THE System SHALL display all generated training examples in a reviewable interface
2. WHEN displaying training examples, THE System SHALL show both the instruction and response for each example
3. WHEN a user selects a training example, THE System SHALL allow editing of both instruction and response text
4. WHEN a user saves edits to a training example, THE System SHALL update the stored example immediately
5. WHEN a user deletes a training example, THE System SHALL remove it from the dataset
6. THE System SHALL display the total count of training examples in the dataset
7. THE System SHALL allow filtering or searching training examples by content

### Requirement 6: JSONL Dataset Export

**User Story:** As a user, I want to download my training dataset in standard format, so that I can use it with external fine-tuning tools or inspect it manually.

#### Acceptance Criteria

1. WHEN a user requests dataset export, THE System SHALL format all training examples as JSONL
2. WHEN formatting JSONL, THE System SHALL include instruction and response fields for each example
3. WHEN formatting JSONL, THE System SHALL ensure each line contains valid JSON
4. THE System SHALL generate a downloadable file with the JSONL content
5. WHEN export is complete, THE System SHALL provide a download link or trigger automatic download
6. THE System SHALL name the exported file with a descriptive name including timestamp

### Requirement 7: Base Model Selection

**User Story:** As a user, I want to choose from supported base models, so that I can fine-tune the model that best fits my requirements and resources.

#### Acceptance Criteria

1. THE System SHALL provide a selection interface for available base models
2. THE System SHALL support at least two base model options for fine-tuning
3. WHEN displaying base models, THE System SHALL show model name, size, and recommended use cases
4. WHEN a user selects a base model, THE System SHALL store the selection for the fine-tuning process
5. THE System SHALL validate that the selected model is compatible with the LoRA fine-tuning pipeline

### Requirement 8: LoRA Fine-Tuning Execution

**User Story:** As a user, I want the system to automatically fine-tune my selected model using my dataset, so that I can create a customized model without manual configuration.

#### Acceptance Criteria

1. WHEN a user initiates fine-tuning, THE System SHALL validate that a dataset and base model are selected
2. WHEN fine-tuning starts, THE System SHALL configure LoRA parameters with sensible defaults
3. WHEN fine-tuning is in progress, THE System SHALL display progress updates to the user
4. WHEN fine-tuning completes successfully, THE System SHALL save the trained adapter weights
5. WHEN fine-tuning fails, THE System SHALL log the error and display a descriptive error message to the user
6. THE System SHALL use QLoRA for memory-efficient training when available
7. WHEN fine-tuning completes, THE System SHALL store metadata about the training run including duration and final loss

### Requirement 9: Model Comparison Interface

**User Story:** As a user, I want to compare responses from the base model and my fine-tuned model, so that I can evaluate the improvement from fine-tuning.

#### Acceptance Criteria

1. THE System SHALL provide a chat interface for testing both base and tuned models
2. WHEN a user enters a prompt, THE System SHALL generate responses from both the base model and tuned model
3. WHEN displaying responses, THE System SHALL clearly label which response came from which model
4. THE System SHALL display both responses side-by-side for easy comparison
5. WHEN generation is in progress, THE System SHALL show loading indicators for each model
6. WHEN a model fails to generate a response, THE System SHALL display an error message for that model only
7. THE System SHALL allow users to enter multiple prompts in sequence for iterative testing

### Requirement 10: Model Deployment and Download

**User Story:** As a user, I want to download my fine-tuned model adapter, so that I can deploy it in my own applications or share it with others.

#### Acceptance Criteria

1. WHEN a user requests adapter download, THE System SHALL package the LoRA adapter weights
2. THE System SHALL include necessary configuration files with the adapter weights
3. WHEN packaging is complete, THE System SHALL provide a download link or trigger automatic download
4. THE System SHALL include instructions or metadata describing how to load the adapter
5. WHEN download fails, THE System SHALL display an error message and allow retry

### Requirement 11: Error Handling and User Feedback

**User Story:** As a user, I want clear feedback when errors occur, so that I understand what went wrong and how to proceed.

#### Acceptance Criteria

1. WHEN any operation fails, THE System SHALL display a user-friendly error message
2. WHEN displaying error messages, THE System SHALL avoid exposing technical stack traces to users
3. WHEN an error is recoverable, THE System SHALL provide actionable guidance for resolution
4. THE System SHALL log all errors with sufficient detail for debugging
5. WHEN a long-running operation is in progress, THE System SHALL provide progress indicators

### Requirement 12: Session State Management

**User Story:** As a user, I want my work to be preserved during my session, so that I don't lose progress if I navigate between different steps.

#### Acceptance Criteria

1. WHEN a user uploads documents, THE System SHALL retain the uploaded files for the duration of the session
2. WHEN a user generates training examples, THE System SHALL store them for the duration of the session
3. WHEN a user edits training examples, THE System SHALL persist the changes within the session
4. WHEN a user completes fine-tuning, THE System SHALL retain access to the tuned model for the duration of the session
5. THE System SHALL allow users to navigate between different pipeline stages without losing data
