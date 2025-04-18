let house_alias, username, password;
let darkmode_tf = false;
let plotsDisplayed = false;
let isRunningLocally = false;
let api_host = isRunningLocally ? 'http://localhost:8000' : 'https://visualizer.electricity.works';

function clearPlots() {
    const plotDivs = [
        'plot1', 'plot2', 'plot3', 'plot4', 'plot5', 
        'plot6', 'plot7', 'plot8', 'plot9', 'plot10', 'plot11',
        'plot-png', 'agg-overview-plot', 'agg-overview-plot2'
    ];
    plotDivs.forEach(plotId => {
        const plotDiv = document.getElementById(plotId);
        if (plotDiv) {
            plotDiv.innerHTML = '';
        }
    });
    const plotContainer = document.getElementById('plot-container');
    if (plotContainer) {
        plotContainer.style.display = 'none';
    }
    const aggOverview = document.getElementById('agg-overview-plot');
    if (!aggOverview) {
        document.getElementById('footer').style.position = 'fixed';
    }
    plotsDisplayed = false;
}

// Check devices's dark mode
const prefersDarkMode = window.matchMedia("(prefers-color-scheme: dark)").matches;
if (prefersDarkMode) {
    document.body.classList.toggle('dark-mode');
    darkmode_tf = prefersDarkMode
}
// Listen for changes in device's dark mode
window.matchMedia("(prefers-color-scheme: dark)").addEventListener('change', (e) => {
    if (e.matches) {
        document.body.classList.add('dark-mode');
        darkmode_tf = true
    } else {
        document.body.classList.remove('dark-mode');
        darkmode_tf = false
    }
    clearPlots();
});

function toggleDarkMode() {
    clearPlots();
    document.body.classList.toggle('dark-mode');
    darkmode_tf = !darkmode_tf;
}

function getDefaultDate(start) {
    const nyDate = new Date(new Date().toLocaleString("en-US", { timeZone: "America/New_York" }));
    if (start) { 
        nyDate.setDate(nyDate.getDate() - 1);
        nyDate.setHours(20, 0, 0, 0);
    } else {
        nyDate.setMinutes(nyDate.getMinutes() + 1);  
    } 
    const year = nyDate.getFullYear();
    const month = String(nyDate.getMonth() + 1).padStart(2, '0');
    const day = String(nyDate.getDate()).padStart(2, '0');
    return `${year}-${month}-${day}`;
}

function getDefaultTime(start) {
    const nyDate = new Date(new Date().toLocaleString("en-US", { timeZone: "America/New_York" }));
    if (start) { 
        nyDate.setDate(nyDate.getDate() - 1);
        nyDate.setHours(20, 0, 0, 0);
    } else {
        nyDate.setMinutes(nyDate.getMinutes() + 1);  
    }  
    const hours = String(nyDate.getHours()).padStart(2, '0');
    const minutes = String(nyDate.getMinutes()).padStart(2, '0');

    return `${hours}:${minutes}`;
}

function setNow() {
    const nyDate = new Date(new Date().toLocaleString("en-US", { timeZone: "America/New_York" }));
    nyDate.setMinutes(nyDate.getMinutes() + 1);
    document.getElementById('end-date-picker').value = nyDate.toISOString().split('T')[0];
    document.getElementById('end-time-picker').value = nyDate.toTimeString().split(' ')[0].substring(0, 5);
    getData(event, false)
}

function setNowAgg() {
    const nyDate = new Date(new Date().toLocaleString("en-US", { timeZone: "America/New_York" }));
    nyDate.setMinutes(nyDate.getMinutes() + 1);
    document.getElementById('end-date-picker').value = nyDate.toISOString().split('T')[0];
    document.getElementById('end-time-picker').value = nyDate.toTimeString().split(' ')[0].substring(0, 5);
    getAggOverviewPlot(event)
}

const aggDateInputs = document.querySelectorAll('.agg-date-input');
aggDateInputs.forEach(input => {
  input.addEventListener('keydown', function(event) {
    if (event.key === 'Enter') {
      getAggOverviewPlot(event);
    }
  });
});

function handleKey(event) {
    if (event.key === "Enter") {
        const target = document.activeElement;
        if (target.tagName !== "BUTTON") {
            event.preventDefault();
            document.querySelector('input[type="submit"]').click();
        }
    }
}

async function LogIn(event) {
    event.preventDefault();
    house_alias = document.getElementById("housealias").value;
    password = document.getElementById("password").value;
    if (house_alias === ""){
        return
    }
    document.getElementById("login-button").style.display = "none";
    try {
        const response = await fetch(`${api_host}/login`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                house_alias: `${house_alias}`,
                password: password, 
            })
        });
        if (response.ok) {
            const data = await response.json();
            if (data === true) {
                document.getElementById("login-div").style.display = "none";
                if (window.innerHeight < 650) {
                    document.getElementById('footer').style.position = 'relative';
                } else {
                    document.getElementById('footer').style.position = 'fixed';
                }
                document.getElementById("data-selector-title").textContent = `${house_alias.charAt(0).toUpperCase()}${house_alias.slice(1)}`;
                document.getElementById('start-date-picker').value = getDefaultDate(true);
                document.getElementById('start-time-picker').value = getDefaultTime(true);
                document.getElementById('end-date-picker').value = getDefaultDate();
                document.getElementById('end-time-picker').value = getDefaultTime();
                document.getElementById("data-selector").style.display = "block";
                getData(event, false);
            } else {
                document.getElementById("housealias").style.border = "1px solid red";
                document.getElementById("password").style.border = "1px solid red";
                document.getElementById("housealias").style.backgroundColor = "rgb(255, 216, 216)";
                document.getElementById("password").style.backgroundColor = "rgb(255, 216, 216)";
                document.getElementById("housealias").value = "";
                document.getElementById("password").value = "";
            }
        }
    } catch (error) {
        console.error('Error trying to log in:', error);
    } finally {
        document.getElementById("login-button").style.display = "block";
    }
}

async function LogInAggregator(event) {
    event.preventDefault();
    username = document.getElementById("username").value;
    password = document.getElementById("password").value;
    document.getElementById("login-button").style.display = "none";
    try {
        const response = await fetch(`${api_host}/login`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                house_alias: `${username}`,
                password: password, 
            })
        });
        if (response.ok) {
            const data = await response.json();
            if (data === true) {
                document.getElementById("login-div").style.display = "none";
                document.getElementById('footer').style.position = 'relative';
                document.getElementById("agg-overview").style.display = "block";
                document.getElementById("price-editor").style.display = "block";
                document.getElementById("agg-overview-title").style.display = "flex";
                document.getElementById("price-editor-title").style.display = "flex";
                document.getElementById('start-date-picker').value = getDefaultDate(true);
                document.getElementById('start-time-picker').value = getDefaultTime(true);
                document.getElementById('end-date-picker').value = getDefaultDate();
                document.getElementById('end-time-picker').value = getDefaultTime();
                initializeTable()
                getAggOverviewPlot(event);
            } else {
                document.getElementById("username").style.border = "1px solid red";
                document.getElementById("password").style.border = "1px solid red";
                document.getElementById("username").style.backgroundColor = "rgb(255, 216, 216)";
                document.getElementById("password").style.backgroundColor = "rgb(255, 216, 216)";
                document.getElementById("username").value = "";
                document.getElementById("password").value = "";
            }
        }
    } catch (error) {
        console.error('Error trying to log in:', error);
    } finally {
        document.getElementById("login-button").style.display = "block";
    }
}

// Re-ordering plots
function movePlotDown(button) {
    const div = button.parentElement.parentElement;
    const nextDiv = div.nextElementSibling;

    if (nextDiv) {
        div.parentElement.insertBefore(nextDiv, div);
    }
}

function movePlotUp(button) {
    const div = button.parentElement.parentElement;
    const prevDiv = div.previousElementSibling;

    if (prevDiv) {
        div.parentElement.insertBefore(div, prevDiv);
    }
}

function dropDownMenu(){
    const navbar = document.getElementById("navbar");
    if (navbar.style.height === '75px') {
        document.getElementById("navbar").style.height = "156.5px";
    } else {
        document.getElementById("navbar").style.height = "75px";
    }
}

function toggleOptions() {
    const checkboxDiv = document.getElementById("options-div");
    if (checkboxDiv.style.display === "none") {
        checkboxDiv.style.display = "block";
        document.getElementById("data-selector").style.display = "none";
        document.getElementById("plot-container").style.display = "none";
        document.getElementById("navbar").style.display = "none";
        document.getElementById('footer').style.position = 'relative';
    } else {
        checkboxDiv.style.display = "none";
        document.getElementById("data-selector").style.display = "block";
        document.getElementById("plot-container").style.display = "block";
        document.getElementById("navbar").style.display = "flex";
        const screenHeight = window.innerHeight;
        if (screenHeight < 650 || plotsDisplayed) {
            document.getElementById('footer').style.position = 'relative';
        } else {
            document.getElementById('footer').style.position = 'fixed';
        }
    }
}

function reorderPlots() {
    const move_buttons = document.getElementsByClassName("move_buttons");
    const reorder_button = document.getElementById("reorder-button");
    for (let i = 0; i < move_buttons.length; i++) {
        if (move_buttons[i].style.display === "none") {
            move_buttons[i].style.display = "block";
            reorder_button.textContent = "Done re-ordering plots";
        } else {
            move_buttons[i].style.display = "none";
            reorder_button.textContent = "Re-order plots";
        }
    }
    toggleOptions();
}

function showLoader() {
    const loader = document.getElementById('loader');
    loader.style.display = 'inline-block';
}

function hideLoader() {
    const loader = document.getElementById('loader');
    loader.style.display = 'none';
}

function enable_button(buttonName) {
    hideLoader()
    let enabledButton;
    if (buttonName === 'plot') {
        enabledButton = document.querySelector('#data-selector-form input[type="submit"]');
        const nowButton = document.getElementById('now-button');
        nowButton.style.display = 'inline';
    } else {
        enabledButton = document.getElementById(`${buttonName}-button`);
    }
    const disabledButton = document.getElementById(`${buttonName}-button-disabled`);
    if (enabledButton) {
        enabledButton.style.display = 'inline';
    }
    if (disabledButton) {
        disabledButton.style.display = 'none';
    }
}

function disable_button(buttonName) {
    // TODO
    const errorText = document.getElementById("error-text")
    if (errorText) {
        errorText.textContent = "";
        errorText.style.display = 'none';
    }
    showLoader()
    let enabledButton;
    if (buttonName === 'plot') {
        enabledButton = document.querySelector('#data-selector-form input[type="submit"]');
        const nowButton = document.getElementById('now-button');
        nowButton.style.display = 'none';
    } else {
        enabledButton = document.getElementById(`${buttonName}-button`);
    }
    const disabledButton = document.getElementById(`${buttonName}-button-disabled`);
    if (enabledButton) {
        enabledButton.style.display = 'none';
    }
    if (disabledButton) {
        disabledButton.style.display = 'inline';
    }
}


async function downloadExcel(event) {
    event.preventDefault();
    disable_button('flo')
    const enddate = document.getElementById('end-date-picker').value;
    const endtime = document.getElementById('end-time-picker').value;
    const endtime_luxon = luxon.DateTime.fromFormat(`${enddate} ${endtime}`, 'yyyy-MM-dd HH:mm', { zone: 'America/New_York' });
    const endUnixMilliseconds = endtime_luxon.toUTC().toMillis();
    try {
        const response = await fetch(`${api_host}/flo`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                house_alias: `${house_alias}`,
                password: password, 
                time_ms: endUnixMilliseconds, 
            })
        });
        if (response.ok) {
            const blob = await response.blob();
            const link = document.createElement('a');
            const url = window.URL.createObjectURL(blob);
            link.href = url;
            const endDateFile = new Date(endUnixMilliseconds);
            const newYorkDate = endDateFile.toLocaleString('en-US', { 
                timeZone: 'America/New_York', 
                hour12: false, 
                year: 'numeric', 
                month: 'short', 
                day: 'numeric', 
                hour: 'numeric', 
            })
            .replace(/,/g, '')
            .replace(/\s/g, '_')
            .toLowerCase();
            link.download = `flo_${house_alias}_${newYorkDate}.xlsx`;
            link.click();
            window.URL.revokeObjectURL(url);
        } else {
            document.getElementById("error-text").textContent = "Error getting FLO";
            document.getElementById("error-text").style.display = 'block'
        }
    } catch (error) {
        console.error('Error getting FLO excel file:', error);
        document.getElementById("error-text").textContent = "Error getting FLO";
        document.getElementById("error-text").style.display = 'block'
    } finally {
        enable_button('flo')
    }
}

async function exportCSV(event, confirmWithUser) {
    event.preventDefault();
    disable_button('csv')
    const startdate = document.getElementById('start-date-picker').value;
    const starttime = document.getElementById('start-time-picker').value;
    const starttime_luxon = luxon.DateTime.fromFormat(`${startdate} ${starttime}`, 'yyyy-MM-dd HH:mm', { zone: 'America/New_York' });
    const startUnixMilliseconds = starttime_luxon.toUTC().toMillis();
    const enddate = document.getElementById('end-date-picker').value;
    const endtime = document.getElementById('end-time-picker').value;
    const endtime_luxon = luxon.DateTime.fromFormat(`${enddate} ${endtime}`, 'yyyy-MM-dd HH:mm', { zone: 'America/New_York' });
    const endUnixMilliseconds = endtime_luxon.toUTC().toMillis();
    const selectedChannels = Array.from(document.querySelectorAll('input[name="channels"]:checked'))
        .map(checkbox => checkbox.value);
    const timestep_wip = document.getElementsByName('csv-timestep')[0];
    const timestep = timestep_wip.value;
    try {
        const response = await fetch(`${api_host}/csv`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                house_alias: `${house_alias}`,
                password: password, 
                start_ms: startUnixMilliseconds, 
                end_ms: endUnixMilliseconds,
                selected_channels: selectedChannels,
                timestep: timestep,
                confirm_with_user: confirmWithUser,
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const contentType = response.headers.get("Content-Type");
        if (contentType && contentType.includes("application/json")) {
            const data = await response.json();
            if ('success' in data && data.success === false) {
                if (data.confirm_with_user) {
                    const userChoice = confirm(data.message);
                    if (userChoice) {
                        await exportCSV(event, true); 
                    }
                } else { 
                    document.getElementById("error-text").textContent = data.message;
                    document.getElementById("error-text").style.display = 'block'
                }
            }
        } else {
            const startDate = new Date(startUnixMilliseconds);
            const formattedStartDate = startDate.toISOString().slice(0, 16).replace('T', '-');
            const endDate = new Date(endUnixMilliseconds);
            const formattedEndDate = endDate.toISOString().slice(0, 16).replace('T', '-');
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${house_alias}_${timestep}s_${formattedStartDate}-${formattedEndDate}.csv`;
            document.body.appendChild(a);
            a.click();
            a.remove();
            window.URL.revokeObjectURL(url);
        }
    } catch (error) {
        console.error('Error getting CSV:', error);
        document.getElementById("error-text").textContent = "Error getting CSV";
        document.getElementById("error-text").style.display = 'block'
    } finally {
        enable_button('csv')
    }
}

async function fetchPlots(house_alias, password, start_ms, end_ms, channels, confirmWithUser) {
    disable_button('plot')
    try {
        const response = await fetch(`${api_host}/plots`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                house_alias: `${house_alias}`,
                password: password, 
                start_ms: start_ms, 
                end_ms: end_ms,
                selected_channels: channels,
                confirm_with_user: confirmWithUser,
                darkmode: darkmode_tf,
            })
        });
        const contentType = response.headers.get("Content-Type");
        if (contentType && contentType.includes("application/json")) {
            const data = await response.json();
            if ('success' in data && data.success === false) {
                if (data.confirm_with_user) {
                    const userChoice = confirm(data.message);
                    if (userChoice) {
                        await fetchPlots(house_alias, password, start_ms, end_ms, channels, true); 
                    }
                } else { 
                    document.getElementById("error-text").textContent = data.message;
                    document.getElementById("error-text").style.display = 'block'
                }
            }
        } else {
            // Function to create move up and down buttons for each plot
            function createMoveButtons() {
                const moveButtonsDiv = document.createElement('div');
                moveButtonsDiv.classList.add('move_buttons');
                moveButtonsDiv.style.display = 'none';
                const moveUpButton = document.createElement('button');
                moveUpButton.classList.add('reorder-arrow');
                moveUpButton.textContent = '⬆︎';
                moveUpButton.onclick = function() {
                    movePlotUp(moveUpButton);
                };
                const moveDownButton = document.createElement('button');
                moveDownButton.classList.add('reorder-arrow');
                moveDownButton.textContent = '⬇︎';
                moveDownButton.onclick = function() {
                    movePlotDown(moveDownButton);
                };
                moveButtonsDiv.appendChild(moveUpButton);
                moveButtonsDiv.appendChild(moveDownButton);
                return moveButtonsDiv;
            }
            const blob = await response.blob();
            const zip = await JSZip.loadAsync(blob);
            clearPlots();
            let iframeCount = 0;
            const plotContainer = document.getElementById('plot-container');
            plotContainer.style.display = 'inline'
            const footer = document.getElementById('footer');
            footer.style.position = 'relative'
            plotsDisplayed = true
            for (const filename of Object.keys(zip.files)) {
                const fileData = await zip.files[filename].async('blob');
                if (filename.endsWith('.html')) {
                    const blob = new Blob([await zip.files[filename].async('text')], { type: 'text/html' });
                    const htmlUrl = URL.createObjectURL(blob);
                    const iframe = document.createElement('iframe');
                    iframe.src = htmlUrl;
                    if (window.innerWidth < 650) {
                        iframe.style.width = '100%';
                    } else {
                        iframe.style.width = '90%';
                        
                    }
                    iframe.style.height = '375px';
                    iframe.style.maxWidth = '1500px';
                    iframe.style.border = 'none';
                    const plotDivs = [
                        document.getElementById('plot1'),
                        document.getElementById('plot2'),
                        document.getElementById('plot3'),
                        document.getElementById('plot4'),
                        document.getElementById('plot5'),
                        document.getElementById('plot6'),
                        document.getElementById('plot7'),
                        document.getElementById('plot8'),
                        document.getElementById('plot9'),
                        document.getElementById('plot10'),
                        document.getElementById('plot11'),
                        document.getElementById('plot-png')
                    ];
                    const currentDiv = plotDivs[iframeCount % plotDivs.length];
                    currentDiv.appendChild(iframe);
                    currentDiv.appendChild(createMoveButtons());
                    iframeCount++;
                } else if (filename.endsWith('.png')) {
                    const imgUrl = URL.createObjectURL(fileData);
                    const img = document.createElement('img');
                    img.src = imgUrl;
                    img.alt = 'Plot Image';
                    plotsDivPng.appendChild(img);
                }
            }
        }
    } catch (error) {
        console.error('Error getting plots:', error);
        document.getElementById("error-text").textContent = "Error getting plots";
        document.getElementById("error-text").style.display = 'block'
    } finally {
        enable_button('plot');
    }
}

async function fetchBids(house_alias, password, start_ms, end_ms, confirmWithUser) {
    disable_button('bid')
    try {
        const response = await fetch(`${api_host}/plots`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                house_alias: `${house_alias}`,
                password: password, 
                start_ms: start_ms, 
                end_ms: end_ms,
                selected_channels: ["bids"],
                confirm_with_user: confirmWithUser,
                darkmode: darkmode_tf,
            })
        });
        const contentType = response.headers.get("Content-Type");
        if (contentType && contentType.includes("application/json")) {
            const data = await response.json();
            if ('success' in data && data.success === false) {
                if (data.confirm_with_user) {
                    const userChoice = confirm(data.message);
                    if (userChoice) {
                        await fetchBids(house_alias, password, start_ms, end_ms, true); 
                    }
                } else { 
                    document.getElementById("error-text").textContent = data.message;
                    document.getElementById("error-text").style.display = 'block'
                }
            }
        } else {
            const blob = await response.blob();
            const zip = await JSZip.loadAsync(blob);
            const plotsDiv = document.getElementById('bids');
            plotsDiv.innerHTML = '';
            const fileNames = Object.keys(zip.files);
            for (const filename of fileNames) {
                const fileData = await zip.files[filename].async('blob');
                const imgUrl = URL.createObjectURL(fileData);
                
                const imageDiv = document.createElement('div');
                const img = document.createElement('img');
                img.src = imgUrl;
                img.alt = 'Plot Image';
                imageDiv.appendChild(img);
                imageDiv.classList.add('bids-container');
                plotsDiv.appendChild(imageDiv);
            }
        }
    } catch (error) {
        console.error('Error getting bids:', error);
        document.getElementById("error-text").textContent = "Error getting bids";
        document.getElementById("error-text").style.display = 'block'
    } finally {
        enable_button('bid');
    }
    }

function getData(event, get_bids) {
    event.preventDefault();
    const selectedChannels = Array.from(document.querySelectorAll('input[name="channels"]:checked'))
        .map(checkbox => checkbox.value);
    const startdate = document.getElementById('start-date-picker').value;
    const starttime = document.getElementById('start-time-picker').value;
    const starttime_luxon = luxon.DateTime.fromFormat(`${startdate} ${starttime}`, 'yyyy-MM-dd HH:mm', { zone: 'America/New_York' });
    const startUnixMilliseconds = starttime_luxon.toUTC().toMillis();
    const enddate = document.getElementById('end-date-picker').value;
    const endtime = document.getElementById('end-time-picker').value;
    const endtime_luxon = luxon.DateTime.fromFormat(`${enddate} ${endtime}`, 'yyyy-MM-dd HH:mm', { zone: 'America/New_York' });
    const endUnixMilliseconds = endtime_luxon.toUTC().toMillis();
    if (get_bids === true) {
        fetchBids(house_alias, password, startUnixMilliseconds, endUnixMilliseconds, false)
    } else {
        fetchPlots(house_alias, password, startUnixMilliseconds, endUnixMilliseconds, selectedChannels, false)
    }
}

async function getAggOverviewPlot(event, just_prices=false) {
    const startdate = document.getElementById('start-date-picker').value;
    const starttime = document.getElementById('start-time-picker').value;
    const starttime_luxon = luxon.DateTime.fromFormat(`${startdate} ${starttime}`, 'yyyy-MM-dd HH:mm', { zone: 'America/New_York' });
    const startUnixMilliseconds = starttime_luxon.toUTC().toMillis();
    const enddate = document.getElementById('end-date-picker').value;
    const endtime = document.getElementById('end-time-picker').value;
    const endtime_luxon = luxon.DateTime.fromFormat(`${enddate} ${endtime}`, 'yyyy-MM-dd HH:mm', { zone: 'America/New_York' });
    const endUnixMilliseconds = endtime_luxon.toUTC().toMillis();

    event.preventDefault();
    let selectedChannelsAgg = []

    disable_button('agg-refresh')
    if (just_prices) {
        document.getElementById('agg-overview-plot2').innerHTML = ''
        selectedChannelsAgg = ['prices']
    } else {
        clearPlots()
    }
    try {
        const response = await fetch(`${api_host}/aggregate-plot`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                house_alias: username,
                password: password,
                start_ms: startUnixMilliseconds,
                end_ms: endUnixMilliseconds,
                selected_channels: selectedChannelsAgg,
                darkmode: darkmode_tf,
            })
        });
        const contentType = response.headers.get("Content-Type");
        if (contentType && contentType.includes("application/json")) {
            const data = await response.json();
            // if ('success' in data && data.success === false) {
            //     document.getElementById("error-text").textContent = data.message;
            //     document.getElementById("error-text").style.display = 'block'
            // }
        } else {
            const blob = await response.blob();
            const zip = await JSZip.loadAsync(blob);
            let iframeCount = 0;
            for (const filename of Object.keys(zip.files)) {
                const fileData = await zip.files[filename].async('blob');
                if (filename.endsWith('.html')) {
                    const blob = new Blob([await zip.files[filename].async('text')], { type: 'text/html' });
                    const htmlUrl = URL.createObjectURL(blob);
                    const iframe = document.createElement('iframe');
                    iframe.src = htmlUrl;
                    if (window.innerWidth < 650) {
                        iframe.style.width = '100%';
                    } else {
                        iframe.style.width = '90%';
                    }
                    iframe.style.maxWidth = '1500px';
                    iframe.style.height = '375px';
                    iframe.style.border = 'none';
                    let plotDivs = [
                        document.getElementById('agg-overview-plot'),
                        document.getElementById('agg-overview-plot2'),
                    ];
                    if (just_prices){
                        plotDivs = [
                            document.getElementById('agg-overview-plot2'),
                        ];
                    }
                    const currentDiv = plotDivs[iframeCount % plotDivs.length];
                    currentDiv.appendChild(iframe);
                    iframeCount++;
                }
            }
        }
    } catch (error) {
        console.error('Error getting aggregate overview plot:', error);
        // document.getElementById("error-text").textContent = "Error getting aggregate overview plot";
        // document.getElementById("error-text").style.display = 'block'
    } finally {
        enable_button('agg-refresh')
    }
}

// --------------------
// For aggregators
// --------------------

let currentLmpList = new Array(48).fill(0.0);
let currentTariffList = new Array(48).fill(0.0);
let lmpList = JSON.parse(JSON.stringify(currentLmpList));
let tariffList = JSON.parse(JSON.stringify(currentTariffList));
let unix_s = [];
let isKeyPressed = false;

async function updateTable() {
    const tableRows = document.querySelectorAll('#price-editor-table tbody tr');
        tableRows.forEach((row, index) => {
            const lmpCell = row.querySelectorAll('.current-value')[0];
            const tariffCell = row.querySelectorAll('.current-value')[1];
            const lmpInput = row.querySelectorAll('input')[0];
            const tariffInput = row.querySelectorAll('input')[1];
            lmpCell.textContent = lmpList[index];
            tariffCell.textContent = tariffList[index];
            updateCellStyle(lmpCell, lmpList[index], currentLmpList[index]);
            updateCellStyle(tariffCell, tariffList[index], currentTariffList[index]);
            lmpInput.placeholder = currentLmpList[index];
            tariffInput.placeholder = currentTariffList[index];
        });
        checkIfPricesChanged();
}

async function getPrices() {
    try {
        const response = await fetch('https://price-forecasts.electricity.works/get_prices', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
        });
        if (!response.ok) {
            throw new Error('Failed to fetch prices');
        }
        const data = await response.json();
        currentLmpList = data.lmp.map(parseFloat);
        currentTariffList = data.dist.map(parseFloat);
        lmpList = data.lmp.map(parseFloat);
        tariffList = data.dist.map(parseFloat);
        updateTable();
    } catch (error) {
        console.error('Error fetching prices:', error);
    }
}

async function getDefaultPrices() {
    try {
        const response = await fetch('https://price-forecasts.electricity.works/get_default_prices', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
        });
        if (!response.ok) {
            throw new Error('Failed to fetch prices');
        }
        const data = await response.json();
        lmpList = data.lmp;
        tariffList = data.dist;
        updateTable();
    } catch (error) {
        console.error('Error fetching default prices:', error);
    }
}

function resetPrices() {
    lmpList = JSON.parse(JSON.stringify(currentLmpList));
    tariffList = JSON.parse(JSON.stringify(currentTariffList));
    updateTable();
}

async function sendPrices(event) {
    event.preventDefault();
    document.getElementById('prices-save-button').style.display = 'block';
    document.getElementById('prices-reset-button').style.display = 'block';
    try {
        const response = await fetch('https://price-forecasts.electricity.works/update_prices', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                unix_s: unix_s,
                lmp: lmpList,
                dist: tariffList
            })
        });
        if (!response.ok) {
            throw new Error('Failed to update prices');
        }
        const data = await response.json();
        currentLmpList = data.lmp.map(parseFloat);
        currentTariffList = data.dist.map(parseFloat);
        lmpList = data.lmp.map(parseFloat);
        tariffList = data.dist.map(parseFloat);
        updateTable();
        getAggOverviewPlot(event, just_prices=true);
    } catch (error) {
        console.error('Error sending prices:', error);
    }
}

function checkIfPricesChanged() {
    function areArraysEqual(arr1, arr2) {
        if (arr1.length !== arr2.length) return false;
        for (let i = 0; i < arr1.length; i++) {
            if (arr1[i] !== arr2[i]) return false;
        }
        return true;
    }
    const allCurrent = areArraysEqual(lmpList, currentLmpList) && areArraysEqual(tariffList, currentTariffList);
    if (allCurrent) {
        document.getElementById('prices-save-button').style.display = 'block';
        document.getElementById('prices-reset-button').style.display = 'block';
    } else {
        document.getElementById('prices-save-button').style.display = 'block';
        document.getElementById('prices-reset-button').style.display = 'block';
    }
}

function setupHourlyRefresh() {
    function scheduleNextRefresh() {
        const now = DateTime.now().setZone('America/New_York');
        const nextHour = now.plus({ hours: 1 }).startOf('hour');
        const millisToNextHour = nextHour.toMillis() - now.toMillis();
        
        console.log(`Next price refresh scheduled for ${nextHour.toFormat('yyyy-MM-dd HH:mm:ss')}`);
        
        setTimeout(() => {
            console.log('Refreshing prices at the top of the hour');
            
            const tableBody = document.querySelector('#price-editor-table tbody');
            tableBody.innerHTML = '';
            
            initializeTable(true).then(() => {
                scheduleNextRefresh();
            }).catch(error => {
                console.error('Error refreshing prices:', error);
                scheduleNextRefresh();
            });
        }, millisToNextHour);
    }
    
    scheduleNextRefresh();
}

const { DateTime } = luxon;

async function initializeTable(isRefresh = false) {
    await getPrices();
    const tableBody = document.querySelector('#price-editor-table tbody');
    tableBody.innerHTML = '';            
    const currentDate = DateTime.now().setZone('America/New_York').startOf('hour');
    unix_s = [];
    
    for (let i = 0; i < 24; i++) {
        let date = currentDate.plus({ hours: i + 1 });                
        unix_s.push(date.toUnixInteger());
        
        // Add row to the table
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${date.toFormat('cccc, LLL dd, yyyy')}</td>
            <td>${date.toFormat('HH:00')}</td>
            <td class="current-value">${currentLmpList[i]}</td>
            <td><input class='price-edit-input' type="text" style="width:50px" placeholder=${currentLmpList[i]}></td>
            <td class="current-value">${currentTariffList[i]}</td>
            <td><input class='price-edit-input' type="text" style="width:50px" placeholder=${currentTariffList[i]}></td>
        `;

        // LMP input event listener
        const lmpInput = row.querySelectorAll('input')[0];
        lmpInput.addEventListener('keydown', function(event) {
            handleArrowKey(event, row, currentLmpList[i], lmpInput, 'lmp');
        });
        lmpInput.addEventListener('blur', function(event) {
            handleBlurEvent(event, row, currentLmpList[i], lmpInput, 'lmp');
        });

        // Tariff input event listener
        const tariffInput = row.querySelectorAll('input')[1];
        tariffInput.addEventListener('keydown', function(event) {
            handleArrowKey(event, row, currentTariffList[i], tariffInput, 'tariff');
        });
        tariffInput.addEventListener('blur', function(event) {
            handleBlurEvent(event, row, currentTariffList[i], tariffInput, 'tariff');
        });
        tableBody.appendChild(row);
    }
    if (!isRefresh) {
        setupHourlyRefresh();
    }
}

function handleBlurEvent(event, row, currentValue, input, type) {
    if (isKeyPressed) {
        isKeyPressed = false;
        return;
    }
    const columnIndex = type === 'lmp' ? 0 : 1;
    const currentCell = row.querySelectorAll('.current-value')[columnIndex];
    let inputValue = input.value;

    // If the input is empty or invalid, revert to the current value
    if (isNaN(inputValue) || inputValue === '') {
        input.value = ''
        return
    }

    // Update the cell with the new value
    currentCell.textContent = inputValue;

    // Update the corresponding list based on the type (lmp or tariff)
    if (type === 'lmp') {
        lmpList[parseInt(row.rowIndex) - 1] = parseFloat(inputValue);
    } else if (type === 'tariff') {
        tariffList[parseInt(row.rowIndex) - 1] = parseFloat(inputValue);
    }

    updateCellStyle(currentCell, inputValue, currentValue);
    input.value = '';
    checkIfPricesChanged();
}

function handleArrowKey(event, row, currentValue, input, type) {
    const columnIndex = type === 'lmp' ? 0 : 1;
    const currentCell = row.querySelectorAll('.current-value')[columnIndex];
    let inputValue = input.value;

    // Handle Enter key
    if (event.key === 'Enter') {
        isKeyPressed = true;
        if ((isNaN(inputValue) || inputValue === '')) {
            inputValue = currentValue;
        }
        currentCell.textContent = inputValue;
        if (type === 'lmp') {
            lmpList[parseInt(row.rowIndex) - 1] = parseFloat(inputValue);
        } else if (type === 'tariff') {
            tariffList[parseInt(row.rowIndex) - 1] = parseFloat(inputValue);
        }

        updateCellStyle(currentCell, inputValue, currentValue);
        input.value = '';
        checkIfPricesChanged();

        const nextRow = row.nextElementSibling;
        if (nextRow) {
            const nextInput = nextRow.querySelectorAll('input')[columnIndex];
            if (nextInput) {
                nextInput.focus();
            }
        }
    }
    // Handle ArrowDown key
    if (event.key === 'ArrowDown') {
        const nextRow = row.nextElementSibling;
        if (nextRow) {
            const nextInput = nextRow.querySelectorAll('input')[columnIndex];
            if (nextInput) {
                nextInput.focus();
            }
        }
    }
    // Handle ArrowUp key
    if (event.key === 'ArrowUp') {
        const prevRow = row.previousElementSibling;
        if (prevRow) {
            const prevInput = prevRow.querySelectorAll('input')[columnIndex];
            if (prevInput) {
                prevInput.focus();
            }
        }
    }
}

function updateCellStyle(cell, inputValue, currentValue) {
    if (parseFloat(inputValue) !== currentValue) {
        cell.style.color = '#466ac4';
        cell.style.fontWeight = 'bold';
    } else {
        cell.style.removeProperty('color');
        cell.style.removeProperty('font-weight');
    }
}
