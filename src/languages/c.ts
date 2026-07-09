import { cModule } from '../generated/c.ts';
import { registerLanguageModule } from '../moduleRegistry.ts';

// Self-registers on import; tables are validated at registration.
registerLanguageModule(cModule);

export { cModule };
