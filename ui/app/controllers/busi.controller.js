(function () {
  "use strict";

  angular.module("inPhaseApp").controller("BusiController", [
    "ApiService",
    function (ApiService) {
      var vm = this;

      vm.loading = true;
      vm.error = null;
      vm.sample = null;
      vm.classCounts = {};
      vm.availableClasses = [];
      vm.selectedClass = "benign";
      vm.requestedIndex = 0;

      vm.hasData = function () {
        return vm.availableClasses.length > 0;
      };

      vm.classTotal = function () {
        return vm.classCounts[vm.selectedClass] || 0;
      };

      vm.selectClass = function (className) {
        vm.selectedClass = className;
        vm.requestedIndex = 0;
        vm.loadSample();
      };

      vm.prev = function () {
        vm.requestedIndex = Math.max(0, vm.requestedIndex - 1);
        vm.loadSample();
      };

      vm.next = function () {
        vm.requestedIndex += 1;
        vm.loadSample();
      };

      vm.jumpTo = function () {
        if (!Number.isFinite(vm.requestedIndex) || vm.requestedIndex < 0) {
          vm.requestedIndex = 0;
        }
        vm.requestedIndex = Math.floor(vm.requestedIndex);
        vm.loadSample();
      };

      vm.loadSample = function () {
        if (!vm.hasData()) {
          vm.loading = false;
          return;
        }

        vm.loading = true;
        vm.error = null;

        ApiService.getBusiSamplePreview(vm.selectedClass, vm.requestedIndex)
          .then(function (data) {
            vm.sample = data;
            vm.requestedIndex = vm.sample.resolved_index;
          })
          .catch(function (error) {
            vm.error = error.detail || "Failed to load BUSI sample preview";
            vm.sample = null;
          })
          .finally(function () {
            vm.loading = false;
          });
      };

      function init() {
        vm.loading = true;
        vm.error = null;

        ApiService.getBusiCounts()
          .then(function (data) {
            vm.classCounts = data || {};
            vm.availableClasses = Object.keys(vm.classCounts).filter(function (className) {
              return vm.classCounts[className] > 0;
            });

            if (vm.availableClasses.length === 0) {
              vm.error = "No BUSI samples found. Place dataset files under data/busi/{class}.";
              vm.sample = null;
              return;
            }

            vm.selectedClass = vm.availableClasses[0];
            vm.requestedIndex = 0;
            vm.loadSample();
          })
          .catch(function (error) {
            vm.loading = false;
            vm.error = error.detail || "Failed to load BUSI class counts";
          });
      }

      init();
    },
  ]);
})();
