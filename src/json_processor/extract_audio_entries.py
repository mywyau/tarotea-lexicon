# def extract_audio_entries(word_entry):
#     audio_entries = []

#     # word audio
#     audio_entries.append({
#         "filename": word_entry["id"],
#         "id": word_entry["id"],
#         "text": word_entry["word"]
#     })

#     # example audios
#     for ex in word_entry.get("examples", []):
#         audio_entries.append({
#             "filename": ex["id"],
#             "id": ex["id"],
#             "text": ex["sentence"]
#         })

#     return audio_entries
