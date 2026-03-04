(function () {
  "use strict";

  angular.module("inPhaseApp").controller("MainController", [
    "$location",
    "$rootScope",
    "ApiService",
    function ($location, $rootScope, ApiService) {
      var vm = this;

      vm.apiBase = ApiService.getBaseUrl();
      vm.pageTitle = "Dashboard";
      vm.apiHealthy = false;
      vm.authenticated = ApiService.isAuthenticated();
      vm.authError = null;
      vm.authLoading = false;
      vm.currentUser = ApiService.getAuthSession();
      vm.loginForm = {
        username: "viewer",
        password: "viewer123",
      };

      vm.isActive = function (path) {
        return $location.path() === path;
      };

      vm.canAccess = function (role) {
        return ApiService.hasRole(role);
      };

      vm.navigate = function (path, requiresAuth, requiredRole, $event) {
        if ($event && typeof $event.preventDefault === "function") {
          $event.preventDefault();
        }
        if (requiresAuth && !vm.authenticated) {
          return;
        }
        if (requiredRole && !vm.canAccess(requiredRole)) {
          return;
        }
        $location.path(path);
      };

      vm.login = function () {
        vm.authLoading = true;
        vm.authError = null;

        ApiService.login(vm.loginForm.username, vm.loginForm.password)
          .then(function () {
            return ApiService.me();
          })
          .then(function (profile) {
            vm.currentUser = profile;
            vm.authenticated = true;
            if (!$location.path()) {
              $location.path("/dashboard");
            }
          })
          .catch(function (error) {
            vm.authenticated = false;
            vm.currentUser = null;
            vm.authError = error.detail || "Login failed";
          })
          .finally(function () {
            vm.authLoading = false;
          });
      };

      vm.logout = function () {
        vm.authLoading = true;
        ApiService.logout()
          .finally(function () {
            vm.authenticated = false;
            vm.currentUser = null;
            vm.authError = null;
            vm.authLoading = false;
            $location.path("/dashboard");
          });
      };

      var pageMap = {
        "/dashboard": "Project Dashboard",
        "/busi": "BUSI Explorer",
        "/industrial": "Industrial Learning Lab",
        "/yolo": "Liver Ultrasound Detection",
        "/yolo-ultrasound": "YOLO Ultrasound Lab",
        "/preprocessing": "Preprocessing Lab",
        "/ndt": "NDT Sample Explorer",
      };

      $rootScope.$on("$routeChangeSuccess", function () {
        vm.pageTitle = pageMap[$location.path()] || "Ultrasound Platform";
      });

      $rootScope.$on("auth:expired", function () {
        vm.authenticated = false;
        vm.currentUser = null;
        vm.authError = "Session expired. Please sign in again.";
      });

      function init() {
        ApiService.health()
          .then(function () {
            vm.apiHealthy = true;
          })
          .catch(function () {
            vm.apiHealthy = false;
          });

        if (!vm.authenticated) {
          return;
        }

        ApiService.me()
          .then(function (profile) {
            vm.currentUser = profile;
            vm.authenticated = true;
          })
          .catch(function (error) {
            vm.authenticated = false;
            vm.currentUser = null;
            vm.authError = error.detail || "Authentication required";
          });
      }

      init();
    },
  ]);
})();
