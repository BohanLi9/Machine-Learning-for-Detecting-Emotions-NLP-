Machine Learning for Detecting Emotions (NLP)


1.Contexte

L'humour, comme la plupart des langages figura"fs, pose des défis linguis"ques intéressants à la NLP, en raison de son accent sur les sens des mots mul"ples, la connaissance culturelle et la compétence pragma"que. L'apprécia"on de l'humour est également un phénomène hautement subjec"f, l'âge, le sexe et le statut socio- économique étant connus pour avoir un impact sur la percep"on d'une blague. Dans ceKe tâche, les textes de fichiers (train.csv et public_dev.csv) sont collecté des é"queKes et des notes auprès d'un ensemble équilibré de groupes d'âge de 18 à 70 ans. Nos annotateurs représentaient également une variété de genres, de posi"ons poli"ques et de niveaux de revenus.
Dans un fichier train.csv donné, il y a un total de six colonnes, qui sont id, text, is_humor, humour_ra"ng, humour_controversy, offense_ra"ng. Toutes les valeurs de la troisième colonne is_humor sont 0 (pas d'humour) ou 1 (humour), indiquant si le texte est humoris"que. Dans la quatrième colonne, humor_ra"ng, nous u"lisons un chiffre pour représenter la subjec"vité de l'apprécia"on de l'humour. Cela vérifiera la différence de niveau d'humour de chaque texte. La cinquième colonne humor_controversy toutes les valeurs sont également 0 ou 1, indiquant que lorsque le texte est classé comme humour, alors prédire si le niveau d'humour provoquera une controverse. La sixième colonne offense_ra"ng indique à quel niveau le texte est offensé pour les u"lisateurs.




2. Exigences et contraintes du projet

Besoins fonc*onnels:
Tâche 1: Il faut neKoyer les textes de fichier (train.csv), puis créer un modèle d'appren"ssage automa"que sur la détec"on de l’humour et l’apprendre.

Tâche 2: Faire des prédic"ons sur le fichier (public_dev.csv).
  Tâche 2a: prédire si le texte sera considéré comme humoris"que. C'est une tâche binaire.
  Tâche 2b: si le texte est classé comme humoris"que, prédire à quel point il est humoris"que. Les valeurs varient entre 0 et 5.
  Tâche 2c: si le texte est classé comme humoris"que, prédire si la note d'humour serait considérée comme controversée. C’est une tâche binaire.

Tâche 3: prédire à quel point un texte est généralement offensant pour les u"lisateurs. Les valeurs varient entre 0 et 5. Ce score a été calculé indépendamment du fait que le texte soit globalement classé comme humoris"que ou offensant.
