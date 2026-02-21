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
      vm.isAdmin = ApiService.hasRole("admin");
      vm.canTrain = ApiService.hasRole("analyst");
      vm.opsSummary = null;
      vm.opsRecent = [];
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

      vm.runTraining = function () {
        if (!vm.canTrain) {
          return;
        }

        vm.trainingLoading = true;
        vm.trainingError = null;

        var payload = {
          include_normal: !!vm.trainingForm.include_normal,
          epochs: Math.max(2, Math.floor(vm.trainingForm.epochs || 12)),
          batch_size: Math.max(4, Math.floor(vm.trainingForm.batch_size || 16)),
          learning_rate: Math.max(0.0001, Number(vm.trainingForm.learning_rate || 0.01)),
        };

        ApiService.runBusiTraining(payload)
          .then(function (response) {
            applyTrainingPayload(response);
          })
          .catch(function (error) {
            vm.trainingError = error.detail || "Failed to run BUSI training";
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

        ApiService.resyncDatasets()
          .then(function (payload) {
            vm.resyncResult = payload;
            load();
          })
          .catch(function (error) {
            vm.resyncError = error.detail || "Failed to resync datasets into database";
          })
          .finally(function () {
            vm.resyncLoading = false;
          });
      };

      function load() {
        vm.loading = true;
        vm.error = null;

        var requests = [
          ApiService.getDashboardSummary(),
          ApiService.getDashboardReadiness(),
          ApiService.listNdtSamples(),
          ApiService.getBusiTrainingLatest(vm.trainingForm.include_normal),
        ];
        if (vm.isAdmin) {
          requests.push(ApiService.getOpsErrorSummary(24 * 60));
          requests.push(ApiService.getOpsErrorRecent(12));
        }

        $q.all(requests)
          .then(function (responses) {
            vm.summary = responses[0];
            vm.readiness = responses[1];
            vm.ndtSamples = responses[2].slice(0, 5);
            applyTrainingPayload(responses[3]);
            if (vm.isAdmin) {
              vm.opsSummary = responses[4];
              vm.opsRecent = responses[5];
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
