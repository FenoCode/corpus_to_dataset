# This script runs the history_to_dataset.py script to create a QA dataset from the history of conversations.

$DatasetPath = "F:\CS524\historybook_to_dataset\dataset" #CHANGE THIS TO YOUR DATASET PATH WITH YOUR DOCUMENTS
$ProcessedDatasetPath = "F:\CS524\historybook_to_dataset\qa_training_dataset" #CHNAGE THIS TO YOUR PROCESSED DATASET PATH
$QA_Script_Output_FilePath = "F:\CS524\historybook_to_dataset\iam-qa-dataset.jsonl" #CHANGE THIS TO YOUR DESIRED OUTPUT FILE PATH
$DatasetFiles = Get-ChildItem -Path $DatasetPath

ForEach ($file in $DatasetFiles)
{
    $filePath = $file.FullName
    Write-Host "Processing file: $filePath" -ForegroundColor Cyan
    python history_to_dataset.py "$filePath" "$QA_Script_Output_FilePath" --keyword-method "tfidf" --model-name "llama3.1"
    
    # Move final result to the processed dataset directory
    $fileOutputPath = Join-Path -Path $ProcessedDatasetPath -ChildPath "temp_$($file.BaseName).jsonl"
    Move-Item -Path $QA_Script_Output_FilePath -Destination $fileOutputPath
    Write-Host "Saved QA pairs to $fileOutputPath" -ForegroundColor Cyan

    # Clean up after run to avoid re-processing of previous chunks
    Remove-Item -Path "./checkpoint.json"
    Remove-Item -Path "./raw_responses.log"
    Remove-Item -Path "./temp_*.jsonl"
    Write-Host "Cleaned up temporary files for next run" -ForegroundColor Yellow

}

$ProcessedFiles = Get-ChildItem -Path $ProcessedDatasetPath

"" | Set-Content -Path $QA_Script_Output_FilePath

ForEach ($file in $ProcessedFiles)
{
    Add-Content $QA_Script_Output_FilePath -Value (Get-Content -Path $file.FullName)
}

Write-Host "All QA dataset pairs located in '$ProcessedDatasetPath' have been merged into '$QA_Script_Output_FilePath'" -ForegroundColor Green