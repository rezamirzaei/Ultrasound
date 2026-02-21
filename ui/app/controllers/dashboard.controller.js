(function () {
  "use strict";

  angular.module("inPhaseApp").controller("DashboardController", [
    "ApiService",
    "$location",
    "$q",
    "$rootScope",
    function (ApiService, $location, $q, $rootScope) {
      var vm = this;

      vm.loading = true;
      vm.error = null;
      vm.summary = null;
      vm.readiness = null;
      vm.ndtSamples = [];
      vm.busiRows = [];
      vm.industrialSummary = null;
      vm.industrialRows = [];
      vm.isAdmin = ApiService.hasRole("admin");
      vm.canTrain = ApiService.hasRole("analyst");
      vm.opsSummary = null;
      vm.opsRecent = [];
      vm.schemaStatus = null;
      vm.schemaError = null;
      vm.clientErrors = ApiService.getClientErrors();
      vm.training = null;
      vm.trainingChart = null;
      vm.trainingLoading = false;
      vm.trainingError = null;
      vm.resyncLoading = false;
      vm.resyncError = null;
      vm.resyncResult = null;
      vm.trainingForm = {
        include_normal: false,
        epochs: 12,
        batch_size: 16,
        learning_rate: 0.01,
      };

      vm.jobs = [];
      vm.jobsLoading = false;
      vm.jobsError = null;
      vm.jobNotice = null;

      vm.busiUpload = {
        class_name: "benign",
        split: "train",
      };
      vm.busiUploadLoading = false;
      vm.busiUploadError = null;
      vm.busiUploadResult = null;

      vm.industrialUpload = {
        dataset_name: "steel_defect",
        split: "train",
        class_name: "crazing",
      };
      vm.industrialUploadLoading = false;
      vm.industrialUploadError = null;
      vm.industrialUploadResult = null;

      vm.go = function (path) {
        $location.path(path);
      };

      vm.refreshClientErrors = function () {
        vm.clientErrors = ApiService.getClientErrors();
      };

      vm.clearClientErrors = function () {
        ApiService.clearClientErrors();
        vm.clientErrors = [];
      };

      vm.hasTrainingCurve = function () {
        return !!(vm.training && vm.training.curve && vm.training.curve.length > 0);
      };

      vm.formatPct = function (value) {
        if (!Number.isFinite(value)) {
          return "n/a";
        }
        return (value * 100).toFixed(1) + "%";
      };

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
          baselineY: yOf(0),
          yGrid: yGrid,
          startEpoch: minEpoch,
          endEpoch: maxEpoch,
          trainPath: buildCurvePath(curve, xOf, yOf, "train_accuracy"),
          testPath: buildCurvePath(curve, xOf, yOf, "test_accuracy"),
          trainPoints: curve.map(function (point) {
            return {
              epoch: point.epoch,
              x: xOf(point.epoch),
              y: yOf(point.train_accuracy),
              value: point.train_accuracy,
            };
          }),
          testPoints: curve.map(function (point) {
            return {
              epoch: point.epoch,
              x: xOf(point.epoch),
              y: yOf(point.test_accuracy),
              value: point.test_accuracy,
            };
          }),
        };
      }

      function applyTrainingPayload(payload) {
        vm.training = payload;
        vm.trainingChart = buildTrainingChart(payload ? payload.curve : []);
      }

      vm.refreshTraining = function () {
        vm.trainingLoading = true;
        vm.trainingError = null;
        ApiService.getBusiTrainingLatest(vm.trainingForm.include_normal)
          .then(function (payload) {
            applyTrainingPayload(payload);
          })
          .catch(function (error) {
            vm.trainingError = error.detail || "Failed to load BUSI training metrics";
          })
          .finally(function () {
            vm.trainingLoading = false;
          });
      };

      vm.refreshJobs = function () {
        if (!vm.canTrain) {
          return;
        }
        vm.jobsLoading = true;
        vm.jobsError = null;
        ApiService.listLearningJobs(20)
          .then(function (jobs) {
            vm.jobs = jobs || [];
          })
          .catch(function (error) {
            vm.jobsError = error.detail || "Failed to load learning jobs";
          })
          .finally(function () {
            vm.jobsLoading = false;
          });
      };

      vm.runTraining = function () {
        if (!vm.canTrain) {
          return;
        }

        vm.trainingLoading = true;
        vm.trainingError = null;
        vm.jobNotice = null;

        var payload = {
          include_normal: !!vm.trainingForm.include_normal,
          epochs: Math.max(2, Math.floor(vm.trainingForm.epochs || 12)),
          batch_size: Math.max(4, Math.floor(vm.trainingForm.batch_size || 16)),
          learning_rate: Math.max(0.0001, Number(vm.trainingForm.learning_rate || 0.01)),
        };

        ApiService.enqueueBusiTrainingJob(payload)
          .then(function (job) {
            vm.jobNotice = "Queued BUSI training job #" + job.job_id;
            vm.refreshJobs();
          })
          .catch(function (error) {
            vm.trainingError = error.detail || "Failed to queue BUSI training";
          })
          .finally(function () {
            vm.trainingLoading = false;
          });
      };

      vm.resyncDatasets = function () {
        if (!vm.isAdmin) {
          return;
        }
        vm.resyncLoading = true;
        vm.resyncError = null;
        vm.resyncResult = null;

        ApiService.enqueueDatasetResyncJob()
          .then(function (job) {
            vm.resyncResult = {
              generated_at: new Date().toISOString(),
              busi_rows_synced: 0,
              ndt_rows_synced: 0,
              industrial_rows_synced: 0,
              job_id: job.job_id,
            };
            vm.jobNotice = "Queued dataset resync job #" + job.job_id;
            vm.refreshJobs();
          })
          .catch(function (error) {
            vm.resyncError = error.detail || "Failed to queue dataset resync";
          })
          .finally(function () {
            vm.resyncLoading = false;
          });
      };

      vm.uploadBusiSample = function () {
        if (!vm.canTrain) {
          return;
        }
        vm.busiUploadLoading = true;
        vm.busiUploadError = null;
        vm.busiUploadResult = null;

        var imageInput = document.getElementById("busi-upload-image");
        var maskInput = document.getElementById("busi-upload-mask");
        if (!imageInput || !imageInput.files || !imageInput.files.length) {
          vm.busiUploadLoading = false;
          vm.busiUploadError = "Select an image file before upload.";
          return;
        }

        var formData = new FormData();
        formData.append("class_name", vm.busiUpload.class_name);
        formData.append("split", vm.busiUpload.split);
        formData.append("image", imageInput.files[0]);
        if (maskInput && maskInput.files && maskInput.files.length) {
          formData.append("mask", maskInput.files[0]);
        }

        ApiService.uploadBusiSample(formData)
          .then(function (payload) {
            vm.busiUploadResult = payload;
            vm.jobNotice = "BUSI sample uploaded to SQL storage";
            imageInput.value = "";
            if (maskInput) {
              maskInput.value = "";
            }
            load();
          })
          .catch(function (error) {
            vm.busiUploadError = error.detail || "BUSI upload failed";
          })
          .finally(function () {
            vm.busiUploadLoading = false;
          });
      };

      vm.uploadIndustrialSample = function () {
        if (!vm.canTrain) {
          return;
        }
        vm.industrialUploadLoading = true;
        vm.industrialUploadError = null;
        vm.industrialUploadResult = null;

        var imageInput = document.getElementById("industrial-upload-image");
        var annotationInput = document.getElementById("industrial-upload-annotation");
        if (!imageInput || !imageInput.files || !imageInput.files.length) {
          vm.industrialUploadLoading = false;
          vm.industrialUploadError = "Select an image file before upload.";
          return;
        }

        var formData = new FormData();
        formData.append("dataset_name", vm.industrialUpload.dataset_name);
        formData.append("split", vm.industrialUpload.split);
        formData.append("class_name", vm.industrialUpload.class_name);
        formData.append("image", imageInput.files[0]);
        if (annotationInput && annotationInput.files && annotationInput.files.length) {
          formData.append("annotation", annotationInput.files[0]);
        }

        ApiService.uploadIndustrialSample(formData)
          .then(function (payload) {
            vm.industrialUploadResult = payload;
            vm.jobNotice = "Industrial sample uploaded to SQL storage";
            imageInput.value = "";
            if (annotationInput) {
              annotationInput.value = "";
            }
            load();
          })
          .catch(function (error) {
            vm.industrialUploadError = error.detail || "Industrial upload failed";
          })
          .finally(function () {
            vm.industrialUploadLoading = false;
          });
      };

      function load() {
        vm.loading = true;
        vm.error = null;

        var requests = [
          ApiService.getDashboardSummary(),
          ApiService.getDashboardReadiness(),
          ApiService.listNdtSamples(),
          ApiService.getIndustrialSummary(),
          ApiService.getBusiTrainingLatest(vm.trainingForm.include_normal),
        ];

        var includeJobs = vm.canTrain;
        var includeAdmin = vm.isAdmin;

        if (includeJobs) {
          requests.push(ApiService.listLearningJobs(20));
        }
        if (includeAdmin) {
          requests.push(ApiService.getOpsErrorSummary(24 * 60));
          requests.push(ApiService.getOpsErrorRecent(12));
          requests.push(ApiService.getDatabaseSchemaStatus());
        }

        $q.all(requests)
          .then(function (responses) {
            var index = 0;
            vm.summary = responses[index++];
            vm.readiness = responses[index++];
            vm.ndtSamples = responses[index++].slice(0, 5);
            vm.industrialSummary = responses[index++];
            vm.industrialRows = (vm.industrialSummary.rows || []).slice(0, 12);
            applyTrainingPayload(responses[index++]);

            if (includeJobs) {
              vm.jobs = responses[index++] || [];
            }
            if (includeAdmin) {
              vm.opsSummary = responses[index++];
              vm.opsRecent = responses[index++];
              vm.schemaStatus = responses[index++];
              vm.schemaError = null;
            }

            vm.busiRows = Object.keys(vm.summary.busi_counts || {})
              .map(function (className) {
                var count = vm.summary.busi_counts[className] || 0;
                var pct = vm.summary.busi_total > 0 ? (count / vm.summary.busi_total) * 100 : 0;
                return {
                  name: className,
                  count: count,
                  pct: pct,
                };
              })
              .sort(function (left, right) {
                return right.count - left.count;
              });
          })
          .catch(function (error) {
            vm.error = error.detail || "Failed to load dashboard summary";
            if (includeAdmin) {
              vm.schemaStatus = null;
              vm.schemaError = error.detail || "Failed to load schema diagnostics";
            }
          })
          .finally(function () {
            vm.loading = false;
          });
      }

      vm.refresh = load;

      $rootScope.$on("api:error", function () {
        vm.refreshClientErrors();
      });

      load();
    },
  ]);
})();
