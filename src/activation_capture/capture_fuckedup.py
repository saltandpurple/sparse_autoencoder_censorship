def save_hook(activations, hook):
    global activation_buffer, row_ix
    activations = activations.detach().cpu()          # [batch, seq, d_model]
    # average over sequence dimension to one vector per prompt
    activations = activations.mean(dim=1)             # [batch, d_model]
    if activation_buffer is None:
        hidden_dim = activations.shape[1]
        activation_buffer = np.memmap(OUTPUT_NPY,
                                      dtype="float32",
                                      mode="w+",
                                      shape=(total, hidden_dim))
    n = activations.shape[0]
    activation_buffer[row_ix:row_ix + n, :] = activations
    activation_buffer.flush()
    row_ix += n

# iterate through stored questions in batches
for offset in tqdm(range(0, total, BATCH_SIZE)):
    current_docs_batch = collection.get(limit=BATCH_SIZE,
                                        offset=offset,
                                        include=["documents"])
    prompts = current_docs_batch["documents"]

    # write mapping once per row for traceability
    for prompt in prompts:
        index_file.write(json.dumps({"row": row_ix, "prompt": prompt}) + "\n")
        row_ix += 1
    index_file.flush()
    row_ix -= len(prompts)  # reset because save_hook increments again

    # tokenise & forward
    tokens = tokenizer(prompts,
                       return_tensors="pt",
                       padding=True,
                       truncation=True,
                       max_length=512).to("cuda")
    with model.hooks([(TARGET_HOOK, save_hook)]):
        _ = model(**tokens)

index_file.close()
print(f"Done. Stored activations in {OUTPUT_NPY}, index in {INDEX_JSONL}")