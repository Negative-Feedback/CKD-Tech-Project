function run() {	//On start up the function will load
			var Home = document.getElementById("Home");
			var About = document.getElementById("About");

			About.style.display = "none";
			Home.style.display = "block";
		}

        function Home_function() {
                    var Home = document.getElementById("Home");
                    var About = document.getElementById("About");

                    if (Home.style.display === "none") {
                        About.style.display = "none";
                        Home.style.display = "block";
                    }
                }
        function About_function() {
                    var Home = document.getElementById("Home");
                    var About = document.getElementById("About");

                    if (About.style.display === "none") {
                        About.style.display = "block";
                        Home.style.display = "none";
                    }
                }