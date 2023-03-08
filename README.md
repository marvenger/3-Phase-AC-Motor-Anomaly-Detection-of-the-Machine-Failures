# 3-Phase-AC-Motor-Anomaly-Detection-of-the-Machine-Failures
Detecting anomaly and alert defects from an unstructured data pool received from current sensors of a 3-phase induction motor at a rate of 10K - 15K instances per second. 


MCSA involves analyzing a motor's electric current signature to detect changes in its characteristics, which can indicate faults such as broken rotor bars, worn bearings and misalignment. The technology is based on the fact that the current drawn by the motor is affected by the condition of the rotor and stator windings as well as the load on the motor. By analyzing the frequency spectrum of the current signal, it is possible to detect characteristic changes that indicate specific types of errors. If the
motor or mechanical parts are faulty, it can be seen from the current waveform. The
's approach here is to record all the peaks (maximum values) and peaks (minimum values) of the waveform.
These points are then investigated and the outliers within them given the probability of a bearing fault.
An important thing to note here is that there is no current signature given that the device is working fine. Therefore, we cannot be sure that changes to the current signature are bugs or other activity. But yes, we can be sure that if the current signature changes, there will of course be some activity.

Dataset: This dataset contains data on three-phase AC motors (3.
2 hp).


Modeling and accuracy estimation were performed using a random forest classifier.

Note that before errors in new data can be predicted, they must be preprocessed, including smoothing and finding the peaks and crests of the smoothed waveforms.



Here's how:

To detect deviations, we need current readings that show the machine is working properly. There is no reference date to compare the data.
There may be many reasons for the change, such as machine mode change, or gear change, etc. So I just detect anomalies in the waveform and give them a value of 1 (which is basically an outlier for the height of the waveform)

I read the file in txt format and then I trace. After drawing them, I observed the pattern of the waves and saw some unusual heights. I am trying to smooth the graph so that there is only one peak or ridge where the anomaly is. After that I calculated all the peaks and highs.
Eventually, in the middle of all those peaks and ridges, I found what I would call an unusual lack or outlier. Then, based on this data, I created random forest classifiers for each stage. Since there are three phases, errors in one phase can be compensated by the other two, but since we don't have a reference reading, I'm just trying to predict anomalies in each phase separately.

So now to predict anomalies on new data, we need to separate the readings for each phase, then smooth them, then find the peaks and troughs, then we can detect anomalies that can be traced back in time or in series at pre-smoothed values
