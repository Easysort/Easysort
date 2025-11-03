# Overvågning af affaldscontainer med Easysort
Easysort er et system, der kan overvåge affaldsfraktioner for urenheder, renhed og overordnet indhold. Systemet anvendes på vores kamera, der opsættes ved containere på genbrugspladser, hvorfra containerer regelmæssigt overvåges.

## Resumé
Easysort overvåger affaldscontainere med kamera og AI for løbende at vurdere indhold, renhed og fejlsortering. Systemet giver gennemsigtighed i, hvad der reelt er i containerne, hvor meget der er af hver fraktion, og hvor ofte der optræder typiske urenheder, samt effekten af ændringer over tid. 

Personalet kan få advarsler ved kritiske fejlsorteringer, og ledelsen får grundlag for rapportering (fx CO2‑besparelser) og prioritering af indsats. Løsningen er designet med GDPR for øje (sløring, kryptering, SSO/2FA), og kan implementeres på få uger via pladstjek, installation/kalibrering og en pilotfase før skalering.

## Nuværende udfordringer ved affaldscontainere
- **Fejlsortering og forurening** driver omkostninger op og forringer materialekvaliteten ved genbrug.
- **Manglende transparens** i fyldningsgrad, reelle indhold og forurening.
- **Svært at drive forandringer** og direkte følge resultaterne. 

## Løsning med Easysort
Easysort kombinerer kamera og AI til at overvåge affaldscontainere og give en omfattende visning af indholdet. Derudover kan systemet give advarsler og alarmere, hvis der er fejlsortering eller forurening, som personalet ønsker at handle på.

Easysort hjælper med at svære på følgende spørgsmål:

- **Overordnet indhold**: Hvad er der i containeren? Hvilke fraktioner er der? Hvor mange af hver fraktion er der?
- **Effekt af ændringer**: Hvordan påvirker ændringer i systemet resultaterne? Kan vi se effekten af ændringer i renhed og forurening?
- **Renhed og forurening**: Hvad er der reelt i containeren? Hvad er de typiske urenheder? Hvor mange urenheder er der?
- **Rapportering**: Hvor meget CO2 kan/har vi sparet på bedre sortering? Hvor skal vi bruge vores ressourcer bedst? 
- (Eventuelt) **Fyldningsgrad**: Hvor fyldt er containeren?

## Aktuel anvendelse
På DTU genbrugsplads overvåges affaldscontainere med Easysort. Her får administrativt personale et overblik over det reelle indhold i containere, og kan se effekten af ændringer i systemet. Derudover bliver personale på pladsen oplyst om fejlsorteringer, som der tidligere er klassificeret som kritiske. 

Dermed sparer genbrugspladsen omkostninger ved eftersorteringer og betydeligt forbedre materialekvaliteten ved genbrug både ved at fjerne fejlsortering og ved diverse tiltag, som de igennem systemet kan se, har haft en positiv effekt.

## Privatlivsbeskyttelse
- GDPR by design: Sløring af personer/biler. Billeder, hvor der kan være personer/biler, opbevaret i en specifik database med udvidet sikkerhed.
- Sikkerhed: Kryptering af data i transit, SSO, 2FA og rollebaseret adgangskontrol.
- Udvidet sikkerhed afhængig af behov.

## Implementeringsplan
1. Pladstjek (uge 0-1): Layout, strøm/PoE, netværk, montage
2. Installation & kalibrering (uge 1-2): Hardware, fraktionsmodeller, alarmtærskler. Løbende kommunikation for at sikre, at systemet finder det, der skal findes.
3. Pilot & Drift (uge 3-6): Baseline, feedback og finjustering. Oprettelse af rapporter og dashboard.

Derfra kan der tages stilling til, om der skal udrulles til flere pladser og/eller containere, hvor det er relevant. 