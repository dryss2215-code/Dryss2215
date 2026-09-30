import certifi; certifi.where=lambda: "/root/.ccr/ca-bundle.crt"
import asyncio, json, edge_tts, subprocess
SCENES = [
 ("hook",  "Stop. Si tu utilises encore ChatGPT comme un simple moteur de recherche, tu passes à côté de l'essentiel."),
 ("cred",  "Ça fait dix ans que je travaille dans l'univers des ebooks. J'en ai lu, analysé et créé des centaines. Et franchement, celui-là sort du lot."),
 ("cover", "Maîtrise ChatGPT en partant de zéro. Un guide clair, sans charabia, pensé pour les vrais débutants."),
 ("list",  "Dedans : la méthode pas à pas pour écrire un prompt qui marche dès le premier essai. Une bibliothèque de prompts prêts à copier-coller. Des prompts spécial études, pour résumer un cours en deux minutes. Et des prompts spécial travail, pour tes mails et tes comptes rendus."),
 ("check", "Et en bonus, la checklist des erreurs à éviter, pour ne plus perdre de temps à tâtonner."),
 ("result","Le résultat ? Tu gagnes plusieurs heures chaque semaine."),
 ("price", "Sa valeur, c'est soixante euros. Aujourd'hui, il est à vingt-neuf euros quatre-vingt-dix-neuf. Paiement unique, accès à vie."),
 ("cta",   "Le lien est dans ma bio. Commence aujourd'hui, ton futur toi te dira merci."),
]
VOICE="fr-FR-HenriNeural"
async def one(i,name,text):
    c=edge_tts.Communicate(text,VOICE,rate="+8%",boundary="WordBoundary")
    words=[]
    with open(f"s{i}.mp3","wb") as f:
        async for ch in c.stream():
            if ch["type"]=="audio": f.write(ch["data"])
            elif ch["type"]=="WordBoundary":
                words.append((ch["offset"]/1e7,(ch["offset"]+ch["duration"])/1e7,ch["text"]))
    subprocess.run(["ffmpeg","-y","-loglevel","error","-i",f"s{i}.mp3","-ar","44100","-ac","1",f"s{i}.wav"],check=True)
    dur=float(subprocess.check_output(["ffprobe","-v","error","-show_entries","format=duration","-of","csv=p=0",f"s{i}.wav"]))
    return dict(name=name,text=text,words=words,dur=dur)
async def main():
    out=[await one(i,n,t) for i,(n,t) in enumerate(SCENES)]
    json.dump(out,open("scenes.json","w"),ensure_ascii=False,indent=1)
    for s in out: print(s["name"],round(s["dur"],2),len(s["words"]))
asyncio.run(main())
