# Intelligent overvågning på omlasteplads
Easysort kan bruges til overvågning af omlasteplads for udvidet modtagekontrol, ballepresser og vejebod.

## Resumé

## Nuværende udfordringer ved genbrugstelte
- **Manuelt arbejde** med at overvåge aktiviteten på omlastepladsen.
- **Aflæsninger** af meget dårligt kvalitet eller forkerte aflæsninger ødelægger kvalitet.
- **** og direkte følge resultaterne. 

## Løsning med Easysort

Easysort overvåger omlastepladsen med kamera og AI for løbende at vurdere indhold, aktivitet og kvalitet. På nuværende tidspunkt har systemet 3 forskellige moduler, der kan anvendes på omlastepladsen:

- **Ballepresser**
- **Vejebod**
- **Modtagekontrol**

### Ballepresser

Her bliver ballepresser overvåget ved både indgang og udgang. Dette hjælper med at besvare spørgsmål som:

- Hvilken fraktion bliver presset og hvornår?
- Hvor effektive har vi været med at få produceret baller?
- Hvor lang tid står vi stille fordelt på forskellige fraktioner? Hvad skyldes dette? 
- Hvor mange baller bliver produceret og hvornår?

Alt information er tilgængelig i realtid (sammen med historiske data) igennem vores dashboard på track.easysort.org. 

### Vejebod

Her bliver vejebod og indgang til pladsen overvåget for at sikre korrekt indvejning og afvejning af alle biler på pladsen. Dette hjælper med at forhindre aflæsninger uden korrekt afvejning, glemt ind- og afvejning af biler, der hurtigt kan være dyrt for en omlasteplads.

### Modtagekontrol

Her bliver aflæsninger til fraktioner overvåget for at sikre korrekt aflæsning af fraktioner. Dette kan hjælpe med at:

1. At frigøre medarbejdere til at arbejde med andre opgaver.
2. Give en indikation af, hvor meget der bliver afleveret og hvornår.
3. Advare ved aflæsninger forkerte steder og/eller fraktioner med forhøjet mængde af urenheder.

## Aktuel anvendelse
Systemet er i dag ved at blive anvendt på Verdis' omlasteplads i Gadstrup. Her bliver systemet allerede anvendt til at overvåge ballepresser, mens de andre 2 moduler er i en indkøringsfase. 

## Privatlivsbeskyttelse
- GDPR by design: Sløring af personer/biler. Billeder, hvor der kan være personer/biler, opbevaret i en specifik database med udvidet sikkerhed.
- Sikkerhed: Kryptering af data i transit, SSO, 2FA og rollebaseret adgangskontrol.
- Udvidet sikkerhed afhængig af behov.

## Implementeringsplan
1. Pladstjek (uge 0-1): Layout, strøm/PoE, netværk, montage, adgang til eksisterende kamera.
2. Installation & kalibrering (uge 1-2): Hardware, fraktionsmodeller, alarmtærskler. Løbende kommunikation for at sikre, at systemet finder det, der skal findes.
3. Pilot & Drift (uge 3-8): Baseline, feedback og finjustering. Oprettelse af rapporter og dashboard.

Derfra kan der tages stilling til, om der skal udrulles til flere pladser og/eller containere, hvor det er relevant. 

## Testfase

Det er muligt at få prøvet systemet af med én af de 3 moduler. Hvis relevante kamera allerede er opsat, kan systemet komme op og køre på mindre end 72 timer fra adgang til pladsen er givet og indlæsning af data er startet.

## Pris

Prisen afhænger af antal kamera og anvendelse. Prisen ligger på nuværende tidspunkt mellem 2.000 - 16.000 kr. pr. måned pr. anvendelse.

Her kan man typisk starte et godt pilotprojekt med eksisterende kamera på pladsen, hvorefter vi kan opsætte yderligere kamera de steder, hvor det er relevant.

Den månedlige ydelse inkluderer drift, support, opdateringer til AI-modeller, analyse i realtid, opbevaring af data og hosting.