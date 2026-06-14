export const PIPELINE_STEPS = [
    { id: 'upload', name: 'File Upload', icon: 'Upload', description: 'Upload files with intent description' },
    { id: 'ocr', name: 'OCR', icon: 'ScanText', description: 'Extract text from documents' },
    { id: 'chunking', name: 'Chunking', icon: 'Layers', description: 'Split text into semantic chunks' },
    { id: 'dataset_gen', name: 'Dataset Gen', icon: 'Database', description: 'Generate synthetic training samples' },
    { id: 'quality_control', name: 'Quality Control', icon: 'ShieldCheck', description: 'Score and filter samples' },
    { id: 'fine_tuning', name: 'Fine-Tuning', icon: 'Brain', description: 'Train model on clean dataset' },
    { id: 'deployment', name: 'Deployment', icon: 'Rocket', description: 'Deploy model via API' },
]

// Which steps apply to each pipeline mode
const STEPS_BY_MODE = {
    dataset_only: ['upload', 'ocr', 'chunking', 'dataset_gen', 'quality_control'],
    finetune_only: ['fine_tuning', 'deployment'],
    dataset_and_finetune: ['upload', 'ocr', 'chunking', 'dataset_gen', 'quality_control', 'fine_tuning'],
    full: ['upload', 'ocr', 'chunking', 'dataset_gen', 'quality_control', 'fine_tuning', 'deployment'],
}

export function getStepsForMode(mode) {
    const stepIds = STEPS_BY_MODE[mode] || STEPS_BY_MODE.full
    return stepIds.map(id => PIPELINE_STEPS.find(s => s.id === id)).filter(Boolean)
}
