# Method figures

`scfr_problem_v7.svg` shows the same current radiograph and generator with and without prior-report text. Only the lower path receives the prior report; it does not depict a prior-image input. `scfr_method_v7.svg` shows three source-valid base reports, two ranker proposals, two finding-verifier proposals, and the five-candidate final router. The matching PNG files are README previews.

The router uses 14 shared text features and five candidate indicators, and returns the selected report unchanged. Its inference inputs are generated reports and valid prior text, with no image pixels, reference reports, or wrong-history generations. Ranker labels .2 and .8 denote FP weights in their fitting targets. Verifier labels 0 and 2 denote nominal token-length allowances relative to current-only; the empty-text fallback can exceed those allowances, as described in the [reproduction guide](../REPRODUCIBILITY.md#cpu-workflow-and-fitting-settings).

Report excerpts are authored schematic examples, not observed model outputs, patient reports, or study records. The deletion/retention example illustrates a sentence operation, not a clinical judgment about the displayed image. The prior-image icon in the method figure is not a second radiograph or a longitudinal image pair.

## Radiograph credit

Both figures embed the same public radiograph: [*Chest Xray PA 3-8-2010.png*](https://commons.wikimedia.org/wiki/File:Chest_Xray_PA_3-8-2010.png), by Stillwaterising, dedicated to the public domain under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). The [source page revision](https://commons.wikimedia.org/w/index.php?title=File:Chest_Xray_PA_3-8-2010.png&oldid=855679536) records the attribution and license.

The figures embed the 1280 x 1038-pixel public preview at proportional scale, without cropping or pixel edits. The image provides illustrative context only and does not establish the truth of the authored report excerpts. No restricted image or participant-level annotation is included.
