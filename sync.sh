#!/bin/bash

# Check if the terminal can parse JSON with jq. If not then install jq
if ! command -v jq &> /dev/null
then
    echo "jq is not installed. Installing now..."
    curl -sS https://webi.sh/jq | bash
fi

# Search for models in the deployment layer
MODEL_DIR="./model"
for MODEL_HASH in $(ls $MODEL_DIR | grep 'model_'); do
    # See if the model has metadata attached
    METADATA_FILE="${MODEL_DIR}/${MODEL_HASH}/metadata.json"

    if [ ! -f "$METADATA_FILE" ]; then
        echo "Error: Metadata file not found at $METADATA_FILE"
        continue
    fi
    # If the model has a metadata file, check external_files to determine which files are needed
    external_files=$(jq -r '.external_files | to_entries[] | "\(.key) \(.value)"' "$METADATA_FILE")
    echo $external_files
    if [ $? -ne 0 ]; then
        echo "Error: Failed to parse JSON from $METADATA_FILE"
        continue
    fi

    while IFS=' ' read -r key url; do
        # If the external file already exists then do nothing
        if [ -f "${MODEL_DIR}/${MODEL_HASH}/${key}" ]; then
            echo "File ${key} already downloaded, skipping..."
            continue
        fi
        # If the file doesn't exist we'll pull from the URL specified
        echo "Downloading ${url} to ${MODEL_DIR}/${MODEL_HASH}/${key}..."
        curl -o "${MODEL_DIR}/${MODEL_HASH}/${key}" "${url}"
        # If the file doesn't download then we'll skip this file and echo the error
        if [ $? -ne 0 ]; then
            echo "Error: Failed to download ${url} to ${MODEL_DIR}/${MODEL_HASH}/${key}"
            continue
        fi
    done <<< "$external_files"
done
