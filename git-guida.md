# Guida pratica ai commit

Riferimento rapido per il lavoro quotidiano con Git. Tutti i comandi si lanciano dal terminale dentro la cartella del repository (`Desktop\Projects`).

## Il ciclo quotidiano

```
git status                        # 1. cosa è cambiato?
git add .                         # 2. metti tutto in scena
git commit -m "descrizione"       # 3. scatta la fotografia
git push                          # 4. manda su GitHub
```

`git status` è il comando che userai più spesso: mostra i file modificati (rosso = non in scena, verde = pronti per il commit). Usalo prima e dopo ogni operazione finché non prendi confidenza.

### Add selettivo

`git add .` prende tutto, ma puoi scegliere cosa includere nel commit:

```
git add cavity_flow/poisson.py    # un file solo
git add cavity_flow/              # una cartella intera
```

Utile quando hai fatto due modifiche scollegate: due `add` + `commit` separati raccontano la storia meglio di un commit unico "modifiche varie".

## Scrivere buoni messaggi

Il messaggio deve dire **cosa** e **perché**, non "aggiornamento". Tra sei mesi ringrazierai.

- Sì: `Corretto segno nel termine convettivo di burgers_1d`
- Sì: `Aggiunto solver Neumann per poisson_2d`
- No: `fix`, `modifiche`, `aggiornato file`

Regola pratica: se nel messaggio scrivi "e" ("corretto X **e** aggiunto Y"), probabilmente erano due commit.

## Guardare la storia

```
git log --oneline                 # storia compatta, un commit per riga
git log                           # storia completa con autore e data
git show d1f02d6                  # cosa è cambiato in quel commit
git diff                          # modifiche non ancora committate
git diff d1f02d6 a3f21c9          # differenze tra due commit
git blame cavity_flow/main.py     # chi ha scritto ogni riga e quando
```

## Annullare le cose (i tre casi tipici)

**1. Ho modificato un file ma voglio tornare all'ultimo commit** (modifica non ancora committata):

```
git restore cavity_flow/main.py
```

Attenzione: la modifica va persa davvero, non è recuperabile.

**2. Ho fatto `git add` ma non voglio più includere quel file nel commit:**

```
git restore --staged cavity_flow/main.py
```

Il file resta modificato, esce solo dalla "scena".

**3. Ho committato ma il commit è sbagliato:**

```
git commit --amend -m "messaggio corretto"    # correggi l'ultimo commit (messaggio o file dimenticati)
git revert d1f02d6                            # crea un commit che annulla quel commit
```

`--amend` solo se NON hai ancora fatto push. Dopo il push, usa `revert`: non riscrive la storia, aggiunge un commit di "annullamento".

## Tornare indietro nel tempo

Per guardare com'era il progetto a un certo commit (senza perdere nulla):

```
git checkout d1f02d6              # ti sposti su quel commit, sola lettura
git checkout master               # torni al presente
```

## Branch: esperimenti senza rischi

Vuoi provare uno schema numerico diverso senza toccare il codice funzionante:

```
git branch esperimento-upwind     # crea il ramo
git checkout esperimento-upwind   # ci entri
... lavori e committi normalmente ...
git checkout master               # torni al codice originale, intatto
git merge esperimento-upwind      # se l'esperimento funziona, lo fondi in master
git branch -d esperimento-upwind  # e cancelli il ramo
```

Se l'esperimento fallisce: torni su master e cancelli il ramo con `git branch -D esperimento-upwind`. Come se niente fosse successo.

## Lavorare con altri

```
git pull                          # scarica i commit degli altri PRIMA di iniziare a lavorare
... lavori, commit ...
git push                          # condividi i tuoi
```

Abitudine d'oro: `pull` a inizio sessione, `push` a fine sessione. Se Git segnala un conflitto, apri il file: le due versioni sono marcate con `<<<<<<<` e `>>>>>>>` — tieni quella giusta, cancella i marcatori, `add` + `commit`.

## Promemoria

- Commit piccoli e frequenti battono commit giganti e rari.
- Committa quando qualcosa **funziona**, così hai sempre un punto di ritorno sicuro.
- Il commit salva in locale; solo `push` manda su GitHub.
- Nel dubbio: `git status` prima, sempre.
