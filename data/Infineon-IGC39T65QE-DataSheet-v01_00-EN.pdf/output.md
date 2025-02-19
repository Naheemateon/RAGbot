# IGC39T65QE High Speed IGBT3 Chip

Features• VCES = 650 V• ICn = 75 A• 650 V trench & field stop technology• High switching speed• Low VCEsat• Low EMI• Low turn-off losses• Positive temperature coefficient

Potential applications

- • Uninterruptible power supplies
- • Welding converters
- • Converters with high switching frequency

Product validation

- • Technology qualified for industrial applications. Ready for validation in industrial applications according to the relevant tests of IEC 60747 and 60749 or alternatively JEDEC47/20/22

Description

- • Recommended for discrete components and modules

|Type|Die size|Delivery form|
|---|---|---|
|IGC39T65QE|6.59 mm x 5.91 mm|Sawn on foil|

Datasheet Please read the sections "Important notice" and "Warnings" at the end of this document

Revision 1.00

www.infineon.com

2023-04-28
# Table of contents

|Description|1|
|---|---|
|Features|1|
|Potential applications|1|
|Product validation|1|
|Table of contents|2|

# 1 Mechanical parameters

# 2 Characteristics

# 3 Chip drawing

# 4 Bare die product specifics

Revision history - 7

Disclaimer - 8

Datasheet

Revision 1.00

2023-04-28
# High Speed IGBT3 Chip

# Mechanical parameters

|Parameter|Values|
|---|---|
|Die size|6.59 mm x 5.91 mm|
|Area total|38.95 mm²|
|Emitter pad size|See chip drawing|
|Gate pad size|See chip drawing|
|Silicon thickness|70 μm|
|Wafer size|200 mm|
|Maximum possible chips per wafer|686|
|Passivation frontside|Photoimide|
|Pad metal|3.2 μm AlSiCu|
|Backside metal|Ni Ag - system|
|Die attach|Electrically conductive epoxy glue and soft solder|
|Frontside interconnect|Wire bond: Al ≤ 500 μm|
|Reject ink dot size (valid for inked delivery form only)|Ø 0.65 mm; max. 1.2 mm|
|Storage environment (<12 months) for original and sealed MBB bags|Ambient atmosphere air, temperature 17°C – 25°C|
|Storage environment (<12 months) for open MBB bags|Acc. to IEC62258-3: Atmosphere >99% Nitrogen or inert gas, Humidity <25%RH, Temperature 17°C – 25°C|

Datasheet Revision 1.00

2023-04-28
# Characteristics

|Parameter|Symbol|Note or test condition|Values|Unit|
|---|---|---|---|---|
|Collector-emitter voltage|VCES|Tvj = 25 °C|650|V|
|DC collector current, limited by Tvjmax|IC|-1)|A| |
|Pulsed collector current, tp limited by Tvjmax|ICpulse|225|A| |
|Gate-emitter voltage|VGE|±20|V| |
|Operating junction temperature|Tvjop|-40...175|°C| |
|Short-circuit withstand time2)|tSC|VCC = 400 V, VGE = 15 V Tvj = 150 °C|5|μs|

1) depending on thermal properties of assembly

2) not subject to production test - verified by design/characterization

3) allowed number of short circuits: &lt;1000; time between short circuits: &gt;1s

# Static characteristics (tested on wafer), Tvj = 25°C

|Parameter|Symbol|Note or test condition|Values|Unit|
|---|---|---|---|---|
|Collector-emitter breakdown voltage|VBRCES|IC = 2 mA, VGE = 0 V|650|V|
|Collector-emitter saturation voltage|VCEsat|VGE = 15 V, IC = 75 A|1.38|V|
|Gate-emitter threshold voltage|VGEth|IC = 1.2 mA, VGE = VCE|4.2|V|
|Zero gate-voltage collector current|ICES|VCE = 650 V, VGE = 0 V|3.8|μA|
|Gate-emitter leakage current|IGES|VCE = 0 V, VGE = 20 V|150|nA|
|Internal gate resistance|RG,int| |none|Ω|

# Electrical characteristics

|Parameter|Symbol|Note or test condition|Values|Unit|
|---|---|---|---|---|
|Collector-emitter saturation voltage|VCEsat|VGE = 15 V, IC = 75 A Tvj = 175 °C|2.3|V|
|Input capacitance|Cies|VCE = 25 V, VGE = 0 V, f = 1000 kHz, Tvj = 25 °C|4620|pF|
|Reverse transfer capacitance|Cres|VCE = 25 V, VGE = 0 V, f = 1000 kHz, Tvj = 25 °C|137|pF|

Datasheet Revision 1.00

2023-04-28
IGC39T65QE

High Speed IGBT3 Chip

3 Chip drawing

Note: In general, from reliability and lifetime point of view, the lower the operating junction temperature and/or the applied voltage, the greater the expected lifetime of any semiconductor device. For "Maximum ratings" and "Electrical characteristics": Not subject to production test, specified by design.

|3 Chip drawing| |
|---|---|
|Die Size|6590 um x 5910 um|
| | |
| |6590|
| |3390|
|426|2774|
|3|E {notch|
|3|8 5|
| |247|
| |2535 1520 mm|
|pad| |
|Emitter pad (in case of multiple emitter pads, all emitter pads connected electrically)| |
|Gate pad| |

Figure 1

Datasheet Revision 1.00

2023-04-28
IGC39T65QE

# High Speed IGBT3 Chip

|Bare die product specifics|Bare die product specifics| |
|---|---|
|• Switching characteristics and thermal properties are dependent on module design and mounting technology and can therefore not be specified for a bare die.| |
|• AQL 0.65 for visual inspection according to failure catalogue.| |
|• Electrostatic discharge sensitive device according to MIL-STD 883.| |
|• Example application: -| |

Datasheet

Revision 1.00

2023-04-28
|Document revision|Date of release|Description of changes|
|---|---|---|
|1.00|2023-04-28|Final datasheet|
|***Legacy Revisions***|***Legacy Revisions***|***Legacy Revisions***|
|V1.0|2012-08-01| |

# Datasheet

Revision 1.00
2023-04-28
# Trademarks

All referenced product or service names and trademarks are the property of their respective owners.

Edition 2023-04-28

Published by Infineon Technologies AG

81726 Munich, Germany

© 2023 Infineon Technologies AG

All Rights Reserved.

Do you have a question about any aspect of this document?

Email: erratum@infineon.com

Document reference IFX-ABG920-001

Important notice

Please note that this product is not qualified according to the AEC Q100 or AEC Q101 documents of the Automotive Electronics Council.

Warnings

Due to technical requirements products may contain dangerous substances. For information on the types in question please contact your nearest Infineon Technologies office.

Except as otherwise explicitly approved by Infineon Technologies in a written document signed by authorized representatives of Infineon Technologies, Infineon Technologies’ products may not be used in any applications where a failure of the product or any consequences of the use thereof can reasonably be expected to result in personal injury.

The information given in this document shall in no event be regarded as a guarantee of conditions or characteristics (“Beschaffenheitsgarantie”). With respect to any examples, hints or any typical values stated herein and/or any information regarding the application of the product, Infineon Technologies hereby disclaims any and all warranties and liabilities of any kind, including without limitation warranties of non-infringement of intellectual property rights of any third party.

In addition, any information given in this document is subject to customer’s compliance with its obligations stated in this document and any applicable legal requirements, norms and standards concerning customer’s products and any use of the product of Infineon Technologies in customer’s applications.

The data contained in this document is exclusively intended for technically trained staff. It is the responsibility of customer’s technical departments to evaluate the suitability of the product for the intended application and the completeness of the product information given in this document with respect to such application.
