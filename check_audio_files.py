import os

folders = [
    "classical",
    "electronic",
    "pop",
    "rock"
]

total = 0

for folder in folders:

    path = os.path.join(".", folder)

    files = [
        file for file in os.listdir(path)
        if file.lower().endswith(".mp3")
    ]

    print(folder, ":", len(files), "MP3 files")

    print("First 10:", files[:10])

    total += len(files)

print("\nTotal MP3 files:", total)