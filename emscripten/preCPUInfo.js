Module = {
  preRun: [function() {
    // Ensure /proc directory exists
    try {
      FS.lookupPath('/proc');
    } catch (e) {
      FS.mkdir('/proc');
    }

    // Read the host's /proc/cpuinfo file
    let cpuinfoContent;
    try {
      if (typeof require !== 'undefined') {
        // Node.js environment: use 'fs' module to read the file
        const fs = require('fs');
        cpuinfoContent = fs.readFileSync('/proc/cpuinfo', 'utf8');
      } else {
        // Non-Node.js environments (e.g., browsers)
        cpuinfoContent = "Mocked CPU info: processor: 0\nvendor_id: GenuineIntel\n";
      }
    } catch (err) {
      console.error("Failed to read /proc/cpuinfo from host:", err);
      cpuinfoContent = "Error: Unable to load /proc/cpuinfo\n";
    }

    // Write the content to the virtual filesystem
    FS.writeFile('/proc/cpuinfo', cpuinfoContent);
  }]
};
