# Image credits

All images are from Wikimedia Commons and are reused under the licence listed below.
Each is downscaled to at most 1100 px and re-encoded as JPEG; no other modification.

| File | Source | Author | Licence |
|---|---|---|---|
| `tabular_medical.jpg` | [Table 3, orthogeriatric interventions](https://commons.wikimedia.org/wiki/File:CIA-109048-T03orthogeriatrics.png) | L. Cortejoso, R. A. Dietz, G. Hofmann, M. Gosch, A. Sattler — *Clinical Interventions in Aging* | CC BY 3.0 |
| `tabular_other.jpg` | [Olga Lundin on Titanic passenger list](https://commons.wikimedia.org/wiki/File:Olga_Lundin_on_Titanic_passenger_list.png) | US Dept. of Commerce and Labor, Bureau of Immigration (1912) | CC0 |
| `signal_medical.jpg` | [12 lead generated sinus rhythm](https://commons.wikimedia.org/wiki/File:12_lead_generated_sinus_rhythm.JPG) | Glenlarson | Public domain |
| `signal_other.jpg` | [C major piano chord waveform](https://commons.wikimedia.org/wiki/File:C-major-piano-chord-waveform.png) | Em3rgent0rdr | CC BY-SA 4.0 |
| `image2d_medical.jpg` | [Normal posteroanterior chest radiograph](https://commons.wikimedia.org/wiki/File:Normal_posteroanterior_(PA)_chest_radiograph_(X-ray).jpg) | Mikael Häggström | CC0 |
| `image2d_other.jpg` | [Crops, Kansas (ASTER, 2001)](https://commons.wikimedia.org/wiki/File:Crops_Kansas_AST_20010624.jpg) | NASA / ASTER | Public domain |
| `volume3d_medical.jpg` | [CT of a normal brain](https://commons.wikimedia.org/wiki/File:CT_of_a_normal_brain_(thumbnail).png) | Mikael Häggström | CC0 |
| `volume3d_other.jpg` | [Ouster OS1-64 lidar point cloud, San Francisco](https://commons.wikimedia.org/wiki/File:Ouster_OS1-64_lidar_point_cloud_of_intersection_of_Folsom_and_Dore_St,_San_Francisco.png) | Daniel L. Lu | CC BY 4.0 |
| `text_medical.jpg` | [Radiology report summarization example, MultiMedBench](https://commons.wikimedia.org/wiki/File:Example_of_the_radiology_report_summarization_task_in_MultiMedBench.png) | Tu et al. | CC BY 4.0 |
| `text_other.jpg` | [The Echo and South Leinster Advertiser, 14 Aug 1920](https://commons.wikimedia.org/wiki/File:Enniscorthy_Echo_-_The_Echo_and_South_Leinster_Advertiser_-_Front_Page_-_Sat_14_Aug_1920.png) | Unknown | Public domain |

Attribution is also carried in the speaker notes of each slide in `../slides.md`.

## Generated figures

`noise.png`, `missingness.png`, `imbalance.png`, `shift.png` and `shortcut.png` are not downloaded.
They are rendered by `scripts/make_figures.py` (`make figures`) from seeded synthetic data, so they rebuild identically anywhere.
Do not edit them by hand; edit the script.
`shortcut.png` is composite: it draws a fake "PORTABLE" marker onto `image2d_medical.jpg` above to show how a spurious cue is learned.
It is a constructed illustration; the underlying phenomenon is Zech et al. 2018.
