(function () {
  "use strict";

  angular.module("inPhaseApp").controller("PreprocessingController", [
    "ApiService",
    function (ApiService) {
      var vm = this;

      vm.classCounts = {};
      vm.availableClasses = [];
      vm.form = {
        class_name: "benign",
        sample_index: 0,
        lambda_tv: 0.06,
        rho: 1.0,
        n_iter: 35,
        clip_limit: 2.5,
      };

      vm.loading = false;
      vm.error = null;
      vm.preview = null;
      vm.sortMetric = "ssim";
      vm.sortedMethods = [];

      vm.presets = {
        fast: {
          lambda_tv: 0.04,
          rho: 1.0,
          n_iter: 12,
          clip_limit: 2.0,
        },
        balanced: {
          lambda_tv: 0.06,
          rho: 1.0,
          n_iter: 35,
          clip_limit: 2.5,
        },
        aggressive: {
          lambda_tv: 0.09,
          rho: 1.2,
          n_iter: 80,
          clip_limit: 3.5,
        },
      };

      vm.maxSampleIndex = function () {
        var total = vm.classCounts[vm.form.class_name] || 0;
        return Math.max(0, total - 1);
      };

      vm.applyPreset = function (presetName) {
        var preset = vm.presets[presetName];
        if (!preset) {
          return;
        }

        vm.form.lambda_tv = preset.lambda_tv;
        vm.form.rho = preset.rho;
        vm.form.n_iter = preset.n_iter;
        vm.form.clip_limit = preset.clip_limit;
      };

      vm.onClassChanged = function () {
        if (vm.form.sample_index > vm.maxSampleIndex()) {
          vm.form.sample_index = vm.maxSampleIndex();
        }
      };

      vm.setSortMetric = function (metric) {
        vm.sortMetric = metric;
        vm.sortedMethods = sortMethods(vm.preview ? vm.preview.methods : [], vm.sortMetric);
      };

      function sortMethods(methods, metric) {
        var cloned = (methods || []).slice();
        return cloned.sort(function (left, right) {
          var leftValue = left.metrics[metric];
          var rightValue = right.metrics[metric];
          if (metric === "rmse" || metric === "cv") {
            return leftValue - rightValue;
          }
          return rightValue - leftValue;
        });
      }

      vm.run = function () {
        vm.onClassChanged();
        vm.loading = true;
        vm.error = null;

        ApiService.previewPreprocessing(vm.form)
          .then(function (response) {
            vm.preview = response.data;
            vm.sortedMethods = sortMethods(vm.preview.methods, vm.sortMetric);
          })
          .catch(function (error) {
            vm.error =
              (error.data && error.data.detail) || "Failed to run preprocessing preview";
          })
          .finally(function () {
            vm.loading = false;
          });
      };

      function init() {
        ApiService.getBusiCounts()
          .then(function (response) {
            vm.classCounts = response.data || {};
            vm.availableClasses = Object.keys(vm.classCounts).filter(function (className) {
              return vm.classCounts[className] > 0;
            });

            if (!vm.availableClasses.length) {
              vm.error = "No BUSI data found. Add images to data/busi/{class} before running previews.";
              return;
            }

            vm.form.class_name = vm.availableClasses[0];
            vm.run();
          })
          .catch(function (error) {
            vm.error = (error.data && error.data.detail) || "Failed to load BUSI class counts";
          });
      }

      init();
    },
  ]);
})();
