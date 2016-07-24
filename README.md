# XBranch
'x42' development commences 23rd July 2016

current stage (24th July 2016) of components:
xRefinery - (Initial) Lab and proving ground (buildsystem setup, for ongoing refinment)
xBoincApi - (Preparing) Adapted/simplified boincapi wrapper/mods, with threadsafety and other robustness enhancements
client    - (Preparing)
taskInspector - (Preparing)

<h1>Seti@home application and tools</h1>
Based on everything learned from development with seti@home (since ~2007):
- modernised cross platform development (Gradle buildsystem)
- reduced debugging through unit and built-in regression testing (reliability, accuracy, precision, confidence metrics)
- plugin archictecture for adaptability (new devices, applications, workloads etc)
- install-time and run-time based dispatch (simplified dynamic optimisation)
- no compromise performance (optimisation), maintainability (cleaner design), AND compatibility (initially Cuda, gradually heterogeneous)
- More intuitive design (options that make sense (to users), rather than arcane code related numbers.)
- Simplify complexity with automation.


