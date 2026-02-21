(function () {
  "use strict";

  angular.module("inPhaseApp").controller("IndustrialController", [
    "ApiService",
    "$q",
    function (ApiService, $q) {
      var vm = this;

      vm.loading = true;
      vm.error = null;
      vm.canTrain = ApiService.hasRole("analyst");

      vm.summary = null;
      vm.datasetRows = [];
      vm.datasetOptions = ["steel_defect", "neu_surface", "casting_defect"];
      vm.selectedDataset = "neu_surface";
      vm.availableSplits = [];
      vm.availableClasses = [];

      vm.sample = null;
      vm.sampleLoading = false;
      vm.sampleError = null;
      vm.sampleRequest = {
        split: "train",
        class_name: "crazing",
        sample_index: 0,
      };

      vm.segmentation = null;
      vm.segmentationLoading = false;
      vm.segmentationError = null;

      vm.training = null;
      vm.trainingChart = null;
      vm.trainingLoading = false;
      vm.trainingError = null;
      vm.jobNotice = null;
      vm.trainingForm = {
        dataset_name: "neu_surface",
        epochs: 12,
        batch_size: 16,
        learning_rate: 0.01,
      };

      vm.jobs = [];
      vm.jobsLoading = false;
      vm.jobsError = null;

      vm.upload = {
        dataset_name: "neu_surface",
        split: "train",
        class_name: "crazing",
      };
      vm.uploadLoading = false;
      vm.uploadError = null;
      vm.uploadResult = null;

      vm.formatPct = function (value) {
        if (!Number.isFinite(value)) {
          return "n/a";
        }
        return (value * 100).toFixed(1) + "%";
      };

      vm.selectDataset = function (datasetName) {
        vm.selectedDataset = datasetName;
        vm.trainingForm.dataset_name = datasetName;
        vm.upload.dataset_name = datasetName;
        syncDatasetSelectors();
        refreshDatasetPanels();
      };

      vm.prevSample = function () {
        vm.sampleRequest.sample_index = Math.max(0, Math.floor(vm.sampleRequest.sample_index || 0) - 1);
        refreshSample();
      };

      vm.nextSample = function () {
        vm.sampleRequest.sample_index = Math.max(0, Math.floor(vm.sampleRequest.sample_index || 0) + 1);
        refreshSample();
      };

      vm.refreshTraining = function () {
        vm.trainingLoading = true;
        vm.trainingError = null;
        ApiService.getIndustrialTrainingLatest(vm.selectedDataset)
          .then(function (payload) {
            vm.training = payload;
            vm.trainingChart = buildTrainingChart(payload ? payload.curve : []);
          })
          .catch(function (error) {
            vm.trainingError = error.detail || "Failed to load industrial learning metrics";
          })
          .finally(function () {
            vm.trainingLoading = false;
          });
      };

      vm.queueTraining = function () {
        if (!vm.canTrain) {
          return;
        }
        vm.trainingLoading = true;
        vm.trainingError = null;
        vm.jobNotice = null;

        var payload = {
          dataset_name: vm.selectedDataset,
          epochs: Math.max(2, Math.floor(vm.trainingForm.epochs || 12)),
          batch_size: Math.max(4, Math.floor(vm.trainingForm.batch_size || 16)),
          learning_rate: Math.max(0.0001, Number(vm.trainingForm.learning_rate || 0.01)),
        };

        ApiService.enqueueIndustrialTrainingJob(payload)
          .then(function (job) {
            vm.jobNotice = "Queued industrial training job #" + job.job_id + " for " + vm.selectedDataset;
            refreshJobs();
          })
          .catch(function (error) {
            vm.trainingError = error.detail || "Failed to queue industrial training";
          })
          .finally(function () {
            vm.trainingLoading = false;
          });
      };

      vm.refreshSampleAndSegmentation = function () {
        refreshSample();
      };

      vm.uploadIndustrialSample = function () {
        if (!vm.canTrain) {
          return;
        }
        vm.uploadLoading = true;
        vm.uploadError = null;
        vm.uploadResult = null;

        var imageInput = document.getElementById("industrial-lab-upload-image");
        var annotationInput = document.getElementById("industrial-lab-upload-annotation");
        if (!imageInput || !imageInput.files || !imageInput.files.length) {
          vm.uploadLoading = false;
          vm.uploadError = "Select an image file before upload.";
          return;
        }

        var formData = new FormData();
        formData.append("dataset_name", vm.upload.dataset_name);
        formData.append("split", vm.upload.split);
        formData.append("class_name", vm.upload.class_name);
        formData.append("image", imageInput.files[0]);
        if (annotationInput && annotationInput.files && annotationInput.files.length) {
          formData.append("annotation", annotationInput.files[0]);
        }

        ApiService.uploadIndustrialSample(formData)
          .then(function (payload) {
            vm.uploadResult = payload;
            vm.jobNotice = "Industrial sample uploaded to SQL storage";
            imageInput.value = "";
            if (annotationInput) {
              annotationInput.value = "";
            }
            loadSummaryOnly();
            refreshSample();
          })
          .catch(function (error) {
            vm.uploadError = error.detail || "Industrial upload failed";
          })
          .finally(function () {
            vm.uploadLoading = false;
          });
      };

      vm.refreshJobs = refreshJobs;

      function buildCurvePath(points, xOf, yOf, valueKey) {
        if (!points || !points.length) {
          return "";
        }
        var d = "";
        points.forEach(function (point, index) {
          var x = xOf(point.epoch);
          var y = yOf(point[valueKey]);
          d += (index === 0 ? "M" : "L") + x.toFixed(2) + " " + y.toFixed(2) + " ";
        });
        return d.trim();
      }

      function buildTrainingChart(curve) {
        if (!curve || !curve.length) {
          return null;
        }

        var width = 680;
        var height = 230;
        var padLeft = 40;
        var padRight = 16;
        var padTop = 16;
        var padBottom = 28;
        var plotWidth = width - padLeft - padRight;
        var plotHeight = height - padTop - padBottom;
        var minEpoch = curve[0].epoch;
        var maxEpoch = curve[curve.length - 1].epoch;
        if (maxEpoch <= minEpoch) {
          maxEpoch = minEpoch + 1;
        }

        function xOf(epoch) {
          return padLeft + ((epoch - minEpoch) / (maxEpoch - minEpoch)) * plotWidth;
        }

        function yOf(value) {
          var clamped = Math.max(0, Math.min(1, value || 0));
          return padTop + (1 - clamped) * plotHeight;
        }

        var yTicks = [0, 0.25, 0.5, 0.75, 1];
        var yGrid = yTicks.map(function (tick) {
          return {
            value: tick,
            y: yOf(tick),
            label: Math.round(tick * 100) + "%",
          };
        });

        return {
          width: width,
          height: height,
          yGrid: yGrid,
          startEpoch: minEpoch,
          endEpoch: maxEpoch,
          trainPath: buildCurvePath(curve, xOf, yOf, "train_accuracy"),
          testPath: buildCurvePath(curve, xOf, yOf, "test_accuracy"),
          trainPoints: curve.map(function (point) {
            return {
              x: xOf(point.epoch),
              y: yOf(point.train_accuracy),
            };
          }),
          testPoints: curve.map(function (point) {
            return {
              x: xOf(point.epoch),
              y: yOf(point.test_accuracy),
            };
          }),
        };
      }

      function uniqueSorted(values) {
        return Array.from(new Set(values || [])).sort();
      }

      function rowsForSelectedDataset() {
        return vm.datasetRows.filter(function (row) {
          return row.dataset_name === vm.selectedDataset;
        });
      }

      function syncDatasetSelectors() {
        var rows = rowsForSelectedDataset();
        vm.availableSplits = uniqueSorted(
          rows.map(function (row) {
            return row.split;
          })
        );
        vm.availableClasses = uniqueSorted(
          rows.map(function (row) {
            return row.class_name;
          })
        );

        if (!vm.availableSplits.length) {
          vm.availableSplits = ["train"];
        }
        if (!vm.availableClasses.length) {
          vm.availableClasses = ["crazing"];
        }

        if (vm.availableSplits.indexOf(vm.sampleRequest.split) < 0) {
          vm.sampleRequest.split = vm.availableSplits[0];
        }
        if (vm.availableClasses.indexOf(vm.sampleRequest.class_name) < 0) {
          vm.sampleRequest.class_name = vm.availableClasses[0];
        }

        if (vm.availableSplits.indexOf(vm.upload.split) < 0) {
          vm.upload.split = vm.availableSplits[0];
        }
        if (vm.availableClasses.indexOf(vm.upload.class_name) < 0) {
          vm.upload.class_name = vm.availableClasses[0];
        }
      }

      function refreshSample() {
        vm.sampleLoading = true;
        vm.sampleError = null;
        vm.segmentationError = null;

        var sampleIndex = Math.max(0, Math.floor(vm.sampleRequest.sample_index || 0));
        vm.sampleRequest.sample_index = sampleIndex;

        $q
          .all([
            ApiService.getIndustrialSamplePreview(
              vm.selectedDataset,
              vm.sampleRequest.split,
              vm.sampleRequest.class_name,
              sampleIndex
            ),
            ApiService.getIndustrialSegmentationPreview(
              vm.selectedDataset,
              vm.sampleRequest.split,
              vm.sampleRequest.class_name,
              sampleIndex
            ),
          ])
          .then(function (responses) {
            vm.sample = responses[0];
            vm.segmentation = responses[1];
            vm.sampleRequest.sample_index = vm.sample.resolved_index;
          })
          .catch(function (error) {
            vm.sample = null;
            vm.segmentation = null;
            vm.sampleError = error.detail || "Failed to load industrial sample preview";
          })
          .finally(function () {
            vm.sampleLoading = false;
          });
      }

      function refreshJobs() {
        if (!vm.canTrain) {
          return;
        }
        vm.jobsLoading = true;
        vm.jobsError = null;
        ApiService.listLearningJobs(30)
          .then(function (jobs) {
            vm.jobs = (jobs || []).filter(function (job) {
              return job.job_type === "industrial_training";
            });
          })
          .catch(function (error) {
            vm.jobsError = error.detail || "Failed to load industrial training jobs";
          })
          .finally(function () {
            vm.jobsLoading = false;
          });
      }

      function refreshDatasetPanels() {
        vm.trainingForm.dataset_name = vm.selectedDataset;
        vm.upload.dataset_name = vm.selectedDataset;
        vm.refreshTraining();
        refreshSample();
        refreshJobs();
      }

      function loadSummaryOnly() {
        return ApiService.getIndustrialSummary().then(function (summary) {
          vm.summary = summary;
          vm.datasetRows = summary.rows || [];
          syncDatasetSelectors();
          if (vm.datasetOptions.indexOf(vm.selectedDataset) < 0) {
            vm.selectedDataset = vm.datasetOptions[0];
          }
        });
      }

      function init() {
        vm.loading = true;
        vm.error = null;
        loadSummaryOnly()
          .then(function () {
            refreshDatasetPanels();
          })
          .catch(function (error) {
            vm.error = error.detail || "Failed to load industrial dataset summary";
          })
          .finally(function () {
            vm.loading = false;
          });
      }

      init();
    },
  ]);
})();
