(function () {
  "use strict";

  angular.module("inPhaseApp").controller("NdtController", [
    "ApiService",
    function (ApiService) {
      var vm = this;

      vm.loading = true;
      vm.detailLoading = false;
      vm.error = null;
      vm.samples = [];
      vm.searchQuery = "";
      vm.selectedName = null;
      vm.selected = null;

      vm.filteredSamples = function () {
        if (!vm.searchQuery) {
          return vm.samples;
        }
        var query = vm.searchQuery.toLowerCase();
        return vm.samples.filter(function (sample) {
          return sample.name.toLowerCase().indexOf(query) !== -1;
        });
      };

      vm.loadDetail = function (sampleName) {
        vm.selectedName = sampleName;
        vm.detailLoading = true;
        vm.error = null;

        ApiService.getNdtSample(sampleName)
          .then(function (response) {
            vm.selected = response.data;
          })
          .catch(function (error) {
            vm.error = (error.data && error.data.detail) || "Failed to load sample details";
          })
          .finally(function () {
            vm.detailLoading = false;
          });
      };

      function init() {
        vm.loading = true;
        vm.error = null;

        ApiService.listNdtSamples()
          .then(function (response) {
            vm.samples = response.data;
            if (vm.samples.length > 0) {
              vm.loadDetail(vm.samples[0].name);
            }
          })
          .catch(function (error) {
            vm.error = (error.data && error.data.detail) || "Failed to load NDT sample list";
          })
          .finally(function () {
            vm.loading = false;
          });
      }

      init();
    },
  ]);
})();
